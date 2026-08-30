#!/usr/bin/env python3
"""Event recorder: which agent launched which step, and what it was given.

inotify on the slurmstepd cgroup tree, so every step is caught — including ones
far shorter than any polling interval. Placement is *static* and exists only
while the step lives (sacct never records which cores or which GPU), so it must
be captured at step start or it is lost forever.

Counters are deliberately NOT recorded here — see slurm_usage.py. IN_DELETE
fires after the cgroup is already gone, so this process cannot read final
counters; the poller's last sample is the source for those. Join on (job, step).

  slurm_ledger.py --out ledger.jsonl [--job JOBID]

Rows:
  {"ev":"start", ts, job, step, agent, cpus, gpus:[{index,uuid}], pids, name}
  {"ev":"end",   ts, job, step, lifetime_s}
"""
import argparse, ctypes, json, os, select, socket, struct, subprocess, sys, time
from pathlib import Path

CG = Path("/sys/fs/cgroup/system.slice/slurmstepd.scope")
# Everything here is node-local: the cgroup only exists on nodes where the step
# has tasks, and SLURM_STEP_GPUS holds that node's indices. A multi-node step
# therefore produces one row per node, and the same GPU index means different
# hardware on different nodes — so rows key on (job, step, node) and the UUID
# is the only globally unique device identity.
NODE = socket.gethostname()
IN_CREATE, IN_DELETE = 0x100, 0x200
# The kernel signals a dropped-event overflow with wd=-1 and this bit. It arrives
# when the queue (default 16384 events) fills because we were not draining it —
# exactly what a long blocking capture can cause. Swallowing it silently would
# turn lost steps into steps that appear never to have existed, so it is loud.
IN_Q_OVERFLOW = 0x4000
CAPTURE_SLEEP = 0.002
EMPTY_GIVEUP  = 30      # ~60 ms with no pids at all => the step is already gone
FULL_TRIES    = 150     # ~300 ms, but only while pids exist and we await the env


def rd(p, default=""):
    try: return Path(p).read_text().strip()
    except OSError: return default


def gpu_uuid_map():
    """SLURM's GPU index is the /dev/nvidiaN minor number, which is NOT
    nvidia-smi's PCI-ordered index — assuming they match mislabels every row."""
    out = {}
    try:
        rows = subprocess.run(["nvidia-smi", "--query-gpu=uuid,pci.bus_id",
                               "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=15).stdout.split("\n")
    except (OSError, subprocess.SubprocessError):
        return out
    by_pci = {}
    for r in rows:
        if "," in r:
            uuid, pci = [x.strip() for x in r.split(",")]
            by_pci[pci.lower()] = uuid
    gdir = Path("/proc/driver/nvidia/gpus")
    if gdir.is_dir():
        for dev in gdir.iterdir():
            minor = None
            for line in rd(dev / "information").splitlines():
                if line.startswith("Device Minor:"):
                    minor = int(line.split(":")[1].strip())
            for pci, uuid in by_pci.items():
                if pci.endswith(dev.name.lower()) or dev.name.lower().endswith(pci):
                    if minor is not None:
                        out[minor] = uuid
    return out


def env_of(pid, key):
    try:
        for kv in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
            if kv.startswith(key.encode() + b"="):
                return kv.split(b"=", 1)[1].decode()
    except OSError:
        pass
    return None


def capture(step_dir, uuids):
    """Race with slurmd: the cgroup appears, then cpuset is set, then the task
    is exec'd with its environment.

    Two very different situations look alike for the first few milliseconds:
    a step that is still starting, and a step that already died. Spending the
    full retry budget on the latter blocks the event loop and can push other
    steps past their own capture window, so bail out early when no pid ever
    appears, and keep retrying only once there is something to wait for."""
    # DO NOT return as soon as cpuset is readable. Measured on this node:
    # user/cpuset.cpus.effective is non-empty at ~6 us but still holds the
    # INHERITED job-wide set (e.g. "0-63"); slurmd narrows it to the step's
    # own cores (e.g. "0-1") only at ~0.85 ms. Emitting early is a WRONG
    # answer, not an early one. This loop re-reads cpuset every iteration and
    # returns only once the environment has landed (~19 ms, when slurmd execs
    # the task), by which point cpuset has long settled.
    for i in range(FULL_TRIES):
        cpus = rd(step_dir / "user/cpuset.cpus.effective") or \
               rd(step_dir / "cpuset.cpus.effective")
        pids = []
        for pf in list(step_dir.glob("user/task_*/cgroup.procs")):
            pids += [int(x) for x in rd(pf).split()]
        if not pids:
            pids = [int(x) for x in rd(step_dir / "cgroup.procs").split()]
        if not pids and i >= EMPTY_GIVEUP:
            return None                      # never populated: already gone
        if cpus and pids:
            gpu = env_of(pids[0], "SLURM_STEP_GPUS")
            agent = env_of(pids[0], "AGENT_ID")
            job = env_of(pids[0], "SLURM_JOB_ID")
            # env may not be readable yet; give it a few more tries
            if gpu is not None or agent is not None or i > 30:
                gpus = [{"index": int(g), "uuid": uuids.get(int(g))}
                        for g in (gpu or "").split(",") if g.isdigit()]
                return {"cpus": cpus, "pids": pids[:16], "gpus": gpus,
                        "agent": agent, "job": job,
                        "name": rd(f"/proc/{pids[0]}/comm"), "tries": i}
        time.sleep(CAPTURE_SLEEP)
    return None


def sluid_of_job(job):
    """SLUID names the cgroup dir for a job. Prefer our own cgroup (free, exact
    when we run inside the job); fall back to asking the controller."""
    try:
        parts = Path("/proc/self/cgroup").read_text().strip().split("/")
        if "slurmstepd.scope" in parts:
            return parts[parts.index("slurmstepd.scope") + 1]
    except (OSError, ValueError, IndexError):
        pass
    try:
        out = subprocess.run(["scontrol", "-d", "show", "step", str(job)],
                             capture_output=True, text=True, timeout=15).stdout
        for line in out.splitlines():
            if "SLUID=" in line:
                return line.split("SLUID=")[1].split()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--job")
    a = ap.parse_args()

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    fd = libc.inotify_init1(0o4000)                       # IN_NONBLOCK
    watched = {}
    # slurmstepd.scope holds EVERY job on the node, other users' included.
    # Watching all of them means running capture() on foreign steps, which both
    # wastes the retry budget and delays capture of our own.
    targets = list(CG.glob("*"))
    if a.job:
        sluid = sluid_of_job(a.job)
        if sluid and (CG / sluid).is_dir():
            targets = [CG / sluid]
    for sluid in targets:
        wd = libc.inotify_add_watch(fd, str(sluid).encode(), IN_CREATE | IN_DELETE)
        if wd >= 0:
            watched[wd] = sluid
    if not watched:
        sys.exit(f"no slurmstepd cgroups under {CG} — is a job running here?")
    uuids = gpu_uuid_map()
    started = {}
    out = open(a.out, "a", buffering=1)
    print(f"watching {len(watched)} job cgroup(s); ledger -> {a.out}", file=sys.stderr)

    while True:
        r, _, _ = select.select([fd], [], [], 1.0)
        if not r:
            continue
        buf = os.read(fd, 16384)
        read_ts = time.time()      # stamp on arrival: capture() below can block
        off = 0
        while off < len(buf):
            wd, mask, _c, ln = struct.unpack_from("iIII", buf, off)
            name = buf[off + 16:off + 16 + ln].split(b"\0")[0].decode()
            off += 16 + ln
            if mask & IN_Q_OVERFLOW:
                out.write(json.dumps({"ev": "overflow", "ts": read_ts,
                                      "node": NODE,
                                      "note": "inotify queue overflowed; "
                                              "an unknown number of step events "
                                              "were dropped by the kernel"}) + "\n")
                print("WARNING: inotify queue overflow — step events were lost",
                      file=sys.stderr)
                continue
            if not name.startswith("step_") or name in ("step_extern", "step_batch"):
                continue
            sluid = watched.get(wd)
            if sluid is None:
                continue
            step = name.replace("step_", "")
            now = read_ts
            if mask & IN_CREATE:
                info = capture(sluid / name, uuids)
                if info is None:
                    out.write(json.dumps({"ev": "start", "ts": now, "node": NODE,
                                          "step": step, "sluid": sluid.name,
                                          "miss": "vanished before capture"}) + "\n")
                    continue
                if a.job and info["job"] != str(a.job):
                    continue
                started[(sluid.name, step)] = now
                out.write(json.dumps({"ev": "start", "ts": now, "node": NODE,
                                      "job": info["job"],
                                      "step": step, "sluid": sluid.name,
                                      "agent": info["agent"], "cpus": info["cpus"],
                                      "gpus": info["gpus"], "pids": info["pids"],
                                      "name": info["name"],
                                      "capture_tries": info["tries"]}) + "\n")
            elif mask & IN_DELETE:
                t0 = started.pop((sluid.name, step), None)
                out.write(json.dumps({"ev": "end", "ts": now, "node": NODE,
                                      "step": step, "sluid": sluid.name,
                                      "lifetime_s": round(now - t0, 3) if t0 else None}) + "\n")


if __name__ == "__main__":
    main()
