#!/usr/bin/env python3
"""Gauge sampler: what each live step is actually using.

The companion to slurm_ledger.py. That records EVENTS (which agent got which
cores and which GPU, captured at step start because placement is unobtainable
afterwards). This records GAUGES, which by nature need sampling.

Join the two offline on (job, step).

  slurm_usage.py --out usage.jsonl [--job JOBID] [--interval 1.0] [--no-gpu]

Rows, one per step per tick:
  {"ts", "job", "step", "agent", "cpu_usec", "cpu_delta_pct", "mem_mb",
   "mem_peak_mb", "nproc", "gpus":[{index,uuid,util_pct,mem_used_mb}]}

Notes on what needs sampling and what does not:
  * cpu.stat usage_usec and memory.peak are monotonic kernel counters, so the
    last sample before a step dies is exact, not an estimate.
  * memory.current and GPU utilisation are instantaneous gauges — a spike
    between ticks is genuinely lost, so the interval is the resolution.
"""
import argparse, ctypes, json, socket, subprocess, sys, time
from pathlib import Path

CG = Path("/sys/fs/cgroup/system.slice/slurmstepd.scope")
NODE = socket.gethostname()      # node-local sampler; merge on (job, step, node)


def rd(p, d=""):
    try: return Path(p).read_text().strip()
    except OSError: return d


def env_of(pid, key):
    try:
        for kv in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
            if kv.startswith(key.encode() + b"="):
                return kv.split(b"=", 1)[1].decode()
    except OSError:
        pass
    return None


class _Util(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Mem(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


class _MemV2(ctypes.Structure):
    """nvmlMemory_v2_t. The v1 struct's `used` is computed as total-free, so it
    silently includes driver/ECC reserved memory — 582 MB per L40S here, which
    made a completely idle GPU look like it held half a gigabyte. v2 reports
    `reserved` separately so `used` means what it says."""
    _fields_ = [("version", ctypes.c_uint), ("total", ctypes.c_ulonglong),
                ("reserved", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


# struct-version handshake NVML uses for versioned APIs
_MEM_V2_VERSION = ctypes.sizeof(_MemV2) | (2 << 24)


class Nvml:
    """Query the driver directly instead of forking nvidia-smi.

    Measured on this node: nvidia-smi costs 82 ms per invocation (8.2% of a core
    at 1 Hz, and ~98% of this sampler's total cost) because every tick forks and
    execs a process. The same numbers via NVML take 0.079 ms for all 8 GPUs —
    about 1000x cheaper. The cost was never the language; it was the fork.
    """

    def __init__(self):
        self.lib = self.handles = None
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() != 0:
                return
            n = ctypes.c_uint()
            lib.nvmlDeviceGetCount_v2(ctypes.byref(n))
            handles = []
            for i in range(n.value):
                h = ctypes.c_void_p()
                if lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) == 0:
                    buf = ctypes.create_string_buffer(96)
                    lib.nvmlDeviceGetUUID(h, buf, 96)
                    handles.append((buf.value.decode(), h))
            self.lib, self.handles = lib, handles
            self.have_v2 = hasattr(lib, "nvmlDeviceGetMemoryInfo_v2")
        except (OSError, AttributeError):
            self.lib = self.handles = None
            self.have_v2 = False

    def snapshot(self):
        if not self.handles:
            return {}
        out = {}
        for uuid, h in self.handles:
            u = _Util()
            ok_u = self.lib.nvmlDeviceGetUtilizationRates(h, ctypes.byref(u)) == 0
            used = total = reserved = None
            if self.have_v2:
                m2 = _MemV2(); m2.version = _MEM_V2_VERSION
                if self.lib.nvmlDeviceGetMemoryInfo_v2(h, ctypes.byref(m2)) == 0:
                    used, total, reserved = m2.used, m2.total, m2.reserved
            if used is None:                       # fall back, minus reserved if known
                m = _Mem()
                if self.lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(m)) == 0:
                    used, total = m.used, m.total
            out[uuid] = {
                "util_pct": float(u.gpu) if ok_u else None,
                "mem_used_mb": round(used / 1048576, 1) if used is not None else None,
                "mem_total_mb": round(total / 1048576, 1) if total is not None else None,
                "mem_reserved_mb": round(reserved / 1048576, 1) if reserved is not None else None,
            }
        return out


def minor_to_uuid():
    """SLURM's GPU index is the /dev/nvidiaN minor number, not nvidia-smi's
    PCI-ordered index; on this node those disagree."""
    m, by_pci = {}, {}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=uuid,pci.bus_id",
                            "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10).stdout
        for line in r.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 2 or not parts[0]:
                continue            # blank / malformed line, e.g. trailing newline
            by_pci[parts[1].lower()] = parts[0]
    except (OSError, subprocess.SubprocessError):
        return m
    g = Path("/proc/driver/nvidia/gpus")
    if g.is_dir():
        for dev in g.iterdir():
            minor = None
            for l in rd(dev / "information").splitlines():
                if l.startswith("Device Minor:"):
                    minor = int(l.split(":")[1].strip())
            for pci, u in by_pci.items():
                if pci.endswith(dev.name.lower()) or dev.name.lower().endswith(pci):
                    if minor is not None:
                        m[minor] = u
    return m


def cpu_usec(d):
    for l in rd(d / "cpu.stat").splitlines():
        if l.startswith("usage_usec"):
            return int(l.split()[1])
    return None


def sample(job, m2u, gpus):
    rows, now = [], time.time()
    for sluid in CG.glob("*"):
        for sd in sluid.glob("step_*"):
            nm = sd.name.replace("step_", "")
            if nm in ("extern", "batch"):
                continue
            pids = []
            for pf in sd.glob("user/task_*/cgroup.procs"):
                pids += [int(x) for x in rd(pf).split()]
            if not pids:
                continue
            jid = env_of(pids[0], "SLURM_JOB_ID")
            if job and jid != str(job):
                continue
            gidx = (env_of(pids[0], "SLURM_STEP_GPUS") or "")
            gl = []
            for g in gidx.split(","):
                if g.isdigit():
                    u = m2u.get(int(g))
                    s = gpus.get(u, {})
                    gl.append({"index": int(g), "uuid": u,
                               "util_pct": s.get("util_pct"),
                               "mem_used_mb": s.get("mem_used_mb"),
                               "mem_reserved_mb": s.get("mem_reserved_mb"),
                               "mem_total_mb": s.get("mem_total_mb")})
            mem = rd(sd / "memory.current") or rd(sd / "user/memory.current")
            peak = rd(sd / "memory.peak") or rd(sd / "user/memory.peak")
            rows.append({"ts": now, "node": NODE, "job": jid, "step": nm,
                         "agent": env_of(pids[0], "AGENT_ID"),
                         "cpu_usec": cpu_usec(sd),
                         "mem_mb": round(int(mem) / 1048576, 1) if mem.isdigit() else None,
                         "mem_peak_mb": round(int(peak) / 1048576, 1) if peak.isdigit() else None,
                         "nproc": len(pids), "gpus": gl})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--job")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    m2u = {} if a.no_gpu else minor_to_uuid()
    nvml = None if a.no_gpu else Nvml()
    prev = {}
    out = open(a.out, "a", buffering=1)
    print(f"sampling every {a.interval}s -> {a.out}", file=sys.stderr)
    while True:
        t0 = time.time()
        gpus = {} if a.no_gpu else nvml.snapshot()
        for r in sample(a.job, m2u, gpus):
            k = (r["job"], r["step"])
            p = prev.get(k)
            if p and r["cpu_usec"] is not None and p[1] is not None:
                dt = r["ts"] - p[0]
                if dt > 0:
                    r["cpu_delta_pct"] = round((r["cpu_usec"] - p[1]) / 1e4 / dt, 1)
            prev[k] = (r["ts"], r["cpu_usec"])
            out.write(json.dumps(r) + "\n")
        time.sleep(max(0, a.interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
