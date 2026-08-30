#!/usr/bin/env python3
"""Render traced runs as a Perfetto timeline.

  visualize.py <run_dir>                    one run
  visualize.py <run_dir> <run_dir> ...      several agents on one combined
                                            timeline, sharing the node

With more than one run it behaves like an Nsight Systems profile: each agent
gets its own lanes, the step lane shows what actually executed and where, and a
flow arrow links the srun the agent issued to the step SLURM eventually ran.
The gap between them is queue wait — the analogue of launch-to-execute latency.

Merges the three streams a run produces:
  agent.jsonl   Claude Code's trajectory  -> inference and tool spans
  ledger.jsonl  step events               -> what each experiment was GIVEN
  usage.jsonl   gauges                    -> what it USED

  visualize.py <run_dir>

Writes <run_dir>/profile.trace.json — Chrome Trace Event format, which
https://ui.perfetto.dev ingests directly (client-side; the file is not uploaded).
"""
import json, re, sys
from datetime import datetime
from pathlib import Path

US = 1_000_000


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def jsonl(p):
    if not p.exists():
        return []
    out = []
    for line in p.open(errors="replace"):
        line = line.strip()
        if line.startswith("{"):
            try: out.append(json.loads(line))
            except json.JSONDecodeError: pass
    return out


def classify(cmd):
    """Label a tool call by intent. NB: a Bash call is not an atomic unit of
    work — one call routinely bundles a zero-cost heredoc write with a compile
    and a benchmark — so this is a hint, not an accounting of cost."""
    tags = []
    if re.search(r"(cat\s*>|tee\s|sed -i|python3?\s*-\s*<<)", cmd): tags.append("edit")
    if re.search(r"\bmake\b|\bgcc\b|\bcargo\b|\bgo build\b", cmd):  tags.append("build")
    if re.search(r"\bsrun\b", cmd):                                 tags.append("srun")
    return " + ".join(tags) if tags else "inspect"


def self_check(ev, run_dir):
    """Assert the things that have actually broken before, so a bad trace is
    reported here instead of looking wrong in someone's Perfetto window."""
    bad = []
    spans = [e for e in ev if e["ph"] == "X"]
    counters = [e for e in ev if e["ph"] == "C"]
    names = {e["args"]["name"] for e in ev
             if e["ph"] == "M" and e["name"] == "thread_name"}

    if not spans:
        bad.append("no spans — nothing will render")
    for e in spans:
        if "dur" not in e or "tid" not in e or "pid" not in e:
            bad.append(f"malformed span {e.get('name')}")
        if e.get("dur", 0) < 0:
            bad.append(f"negative duration on {e.get('name')}")
        if e.get("ts", 0) < 0:
            bad.append(f"negative timestamp on {e.get('name')} — clock origin is wrong")
    for e in counters:
        a = e.get("args", {})
        if not a or not all(isinstance(v, (int, float)) for v in a.values()):
            bad.append(f"non-numeric counter {e.get('name')}")

    # every span track must be declared, or Perfetto shows a bare tid
    declared = {e["tid"] for e in ev if e["ph"] == "M" and e["name"] == "thread_name"}
    for e in spans:
        if e["tid"] not in declared:
            bad.append(f"span on undeclared track tid={e['tid']}")
            break

    # the streams must share a clock, or the lanes will not line up
    if counters and spans:
        c0 = min(e["ts"] for e in counters)
        c1 = max(e["ts"] for e in counters)
        s1 = max(e["ts"] + e.get("dur", 0) for e in spans)
        if c0 > s1 or c1 < min(e["ts"] for e in spans):
            bad.append("counter and span clocks do not overlap")

    if not any("Experiment steps" in n for n in names):
        bad.append("no experiment-step lane — did the agent run anything under srun?")
    return bad


def build(run_dir):
    run_dir = Path(run_dir)
    agent = jsonl(run_dir / "agent.jsonl")
    ledger = jsonl(run_dir / "ledger.jsonl")
    usage = jsonl(run_dir / "usage.jsonl")

    # One clock for everything, taken from the runner's own start stamp.
    t0f = run_dir / "t0.epoch"
    origins = []
    if t0f.exists():
        try: origins.append(float(t0f.read_text().strip()))
        except ValueError: pass
    for r in ledger: origins.append(r["ts"])
    for r in usage:  origins.append(r["ts"])

    issued, returned = {}, {}
    for e in agent:
        t = e.get("timestamp")
        if e.get("type") == "assistant" and t:
            for c in e["message"].get("content", []):
                if c.get("type") == "tool_use":
                    issued[c["id"]] = (ts(t), c["name"], c["input"].get("command", ""))
        if e.get("type") == "user" and t:
            for c in (e.get("message", {}).get("content") or []):
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    returned[c["tool_use_id"]] = ts(t)
    calls = sorted((t, returned[i], n, cmd) for i, (t, n, cmd) in issued.items() if i in returned)
    origins += [c[0] for c in calls]
    if not origins:
        sys.exit(f"nothing to render in {run_dir}")
    T0 = min(origins)

    ev = [{"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "Traced run"}}]
    tid_of, next_tid = {}, [1]

    def track(label):
        if label not in tid_of:
            tid_of[label] = next_tid[0]
            ev.append({"ph": "M", "pid": 1, "tid": next_tid[0],
                       "name": "thread_name", "args": {"name": label}})
            next_tid[0] += 1
        return tid_of[label]

    # agent: inference gaps and tool calls
    if calls:
        t_tools, t_think = track("Agent tool calls"), track("Agent inference")
        for a, b, name, cmd in calls:
            ev.append({"ph": "X", "pid": 1, "tid": t_tools, "cat": classify(cmd),
                       "name": f"{name} — {classify(cmd)}",
                       "ts": (a - T0) * US, "dur": (b - a) * US,
                       "args": {"command": cmd[:2000], "wall seconds": round(b - a, 2)}})
        for i in range(len(calls) - 1):
            g0, g1 = calls[i][1], calls[i + 1][0]
            if g1 > g0:
                ev.append({"ph": "X", "pid": 1, "tid": t_think, "cat": "inference",
                           "name": f"thinking {g1-g0:.0f}s",
                           "ts": (g0 - T0) * US, "dur": (g1 - g0) * US,
                           "args": {"wall seconds": round(g1 - g0, 1)}})

    # experiments: one span per step, from the ledger's start/end pair
    starts = {}
    for r in ledger:
        key = (r.get("node"), r.get("step"))
        if r["ev"] == "start":
            starts[key] = r
        elif r["ev"] == "end" and key in starts:
            s = starts.pop(key)
            if s.get("miss"):
                continue
            gl = s.get("gpus") or []
            lane = f"Experiment steps — {s.get('agent') or 'untagged'}"
            ev.append({"ph": "X", "pid": 1, "tid": track(lane), "cat": "step",
                       "name": f"step {s['step']}  cores {s.get('cpus')}"
                               + (f"  gpu {','.join(str(g['index']) for g in gl)}" if gl else ""),
                       "ts": (s["ts"] - T0) * US, "dur": (r["ts"] - s["ts"]) * US,
                       "args": {"agent": s.get("agent"), "cores": s.get("cpus"),
                                "gpu uuids": [g.get("uuid") for g in gl],
                                "lifetime seconds": r.get("lifetime_s")}})
    for key, s in starts.items():                       # still running at the end
        if not s.get("miss"):
            ev.append({"ph": "X", "pid": 1,
                       "tid": track(f"Experiment steps — {s.get('agent') or 'untagged'}"),
                       "cat": "step", "name": f"step {s['step']} (unterminated)",
                       "ts": (s["ts"] - T0) * US, "dur": 1 * US, "args": {}})

    # gauges: counters, one series per step so lanes stay separable
    for r in usage:
        base = {"pid": 1, "ts": (r["ts"] - T0) * US, "ph": "C"}
        who = r.get("agent") or f"step {r.get('step')}"
        if r.get("cpu_delta_pct") is not None:
            ev.append({**base, "name": "CPU utilization %", "args": {who: r["cpu_delta_pct"]}})
        if r.get("mem_mb") is not None:
            ev.append({**base, "name": "Resident memory MB", "args": {who: r["mem_mb"]}})
        for g in r.get("gpus") or []:
            if g.get("util_pct") is not None:
                ev.append({**base, "name": "GPU utilization %",
                           "args": {f"gpu {g['index']}": g["util_pct"]}})

    problems = self_check(ev, run_dir)
    out = run_dir / "profile.trace.json"
    out.write_text(json.dumps({"traceEvents": ev, "displayTimeUnit": "ms"}))

    # summary
    span = max((e["ts"] + e.get("dur", 0)) / US for e in ev if e["ph"] == "X") if calls or ledger else 0
    steps = sum(1 for e in ev if e.get("cat") == "step")
    misses = sum(1 for r in ledger if r.get("miss"))
    reward = "n/a"
    rf = run_dir / "result.json"
    if rf.exists():
        try: reward = json.loads(rf.read_text()).get("reward")
        except Exception: pass
    print(f"=== {run_dir.name} ===")
    print(f"  span            {span:8.1f}s")
    print(f"  agent tool calls{len(calls):8}")
    print(f"  experiment steps{steps:8}   (misses: {misses})")
    print(f"  usage samples   {len(usage):8}")

    # The ledger is job-wide, so a run's trace shows every agent sharing the
    # node. That is the point: it is how contention between concurrent agents
    # becomes visible rather than inferred.
    per = {}
    for r in ledger:
        if r["ev"] == "start" and not r.get("miss"):
            a = per.setdefault(r.get("agent") or "untagged", {"steps": 0, "cores": 0, "gpus": 0})
            a["steps"] += 1
            c = r.get("cpus") or ""
            n = 0
            for part in c.split(","):
                if "-" in part:
                    lo, hi = part.split("-"); n += int(hi) - int(lo) + 1
                elif part:
                    n += 1
            a["cores"] += n
            a["gpus"] += len(r.get("gpus") or [])
    if len(per) > 1 or (per and "untagged" not in per):
        print(f"\n  by agent:")
        print(f"    {'agent':<28} {'steps':>6} {'core-steps':>11} {'gpu-steps':>10}")
        for a, v in sorted(per.items(), key=lambda kv: -kv[1]["steps"]):
            print(f"    {a:<28} {v['steps']:>6} {v['cores']:>11} {v['gpus']:>10}")
    print(f"  reward          {reward}")
    lanes = [e["args"]["name"] for e in ev
             if e["ph"] == "M" and e["name"] == "thread_name"]
    print(f"  lanes           {', '.join(lanes)}")
    if problems:
        print("\n  SELF-CHECK FAILED:")
        for b in problems:
            print(f"    - {b}")
    else:
        print("\n  self-check: spans well-formed, tracks declared, clocks aligned")
    # a copy at the repo root, where an editor file tree can reach it
    repo = Path(__file__).resolve().parent.parent
    tdir = repo / "traces"
    try:
        tdir.mkdir(exist_ok=True)
        gi = tdir / ".gitignore"
        if not gi.exists():
            gi.write_text("*\n!.gitignore\n")
        (tdir / f"{run_dir.name}.trace.json").write_text(out.read_text())
        print(f"  copy:  traces/{run_dir.name}.trace.json")
    except OSError:
        pass
    print(f"\n  trace: {out}")
    print("  open at https://ui.perfetto.dev  (Open trace file)")


def combine(run_dirs, out_path=None):
    """One timeline across several agents sharing a node."""
    runs = [Path(d) for d in run_dirs]
    ledger, usage = [], []
    seen = set()
    for d in runs:                       # snapshots are job-wide; de-duplicate
        for r in jsonl(d / "ledger.jsonl"):
            k = (r.get("node"), r.get("step"), r.get("ev"), round(r.get("ts", 0), 3))
            if k not in seen:
                seen.add(k); ledger.append(r)
        for r in jsonl(d / "usage.jsonl"):
            k = (r.get("node"), r.get("step"), round(r.get("ts", 0), 2))
            if k not in seen:
                seen.add(k); usage.append(r)

    agents = {}                          # agent tag -> (run dir, tool calls)
    for d in runs:
        issued, returned = {}, {}
        for e in jsonl(d / "agent.jsonl"):
            t = e.get("timestamp")
            if e.get("type") == "assistant" and t:
                for c in e["message"].get("content", []):
                    if c.get("type") == "tool_use":
                        issued[c["id"]] = (ts(t), c["name"], c["input"].get("command", ""))
            if e.get("type") == "user" and t:
                for c in (e.get("message", {}).get("content") or []):
                    if isinstance(c, dict) and c.get("type") == "tool_result":
                        returned[c["tool_use_id"]] = ts(t)
        calls = sorted((a, returned[i], n, cmd)
                       for i, (a, n, cmd) in issued.items() if i in returned)
        tag = None
        for r in ledger:                  # the tag this run's steps carry
            if r.get("agent") and d.name.endswith(str(r["agent"]).split("-")[-1]):
                tag = r["agent"]; break
        agents[tag or d.name] = (d, calls)

    origins = [c[0] for _, c in agents.values() for c in [c[0]] if c] + \
              [r["ts"] for r in ledger] + [r["ts"] for r in usage]
    origins = [o for o in origins if o]
    if not origins:
        sys.exit("nothing to render")
    T0 = min(origins)

    ev = [{"ph": "M", "pid": 0, "name": "process_name", "args": {"name": "Node"}}]
    pid = {}
    flow_id = [0]
    unmatched = [0]

    def lane(p, label):
        key = (p, label)
        if key not in lane.ids:
            lane.ids[key] = len(lane.ids) + 1
            ev.append({"ph": "M", "pid": p, "tid": lane.ids[key],
                       "name": "thread_name", "args": {"name": label}})
        return lane.ids[key]
    lane.ids = {}

    for i, (tag, (d, calls)) in enumerate(sorted(agents.items()), start=1):
        pid[tag] = i
        ev.append({"ph": "M", "pid": i, "name": "process_name",
                   "args": {"name": f"{tag}  ({d.name})"}})
        t_tool, t_think = lane(i, "tool calls"), lane(i, "inference")
        launches = []
        for a, b, name, cmd in calls:
            is_srun = "srun" in cmd
            ev.append({"ph": "X", "pid": i, "tid": t_tool,
                       "cat": "srun" if is_srun else classify(cmd),
                       "name": ("srun launch" if is_srun else f"{name} — {classify(cmd)}"),
                       "ts": (a - T0) * US, "dur": (b - a) * US,
                       "args": {"command": cmd[:2000], "wall seconds": round(b - a, 2)}})
            if is_srun:
                launches.append((a, b, cmd))
        for j in range(len(calls) - 1):
            g0, g1 = calls[j][1], calls[j + 1][0]
            if g1 > g0:
                ev.append({"ph": "X", "pid": i, "tid": t_think, "cat": "inference",
                           "name": f"thinking {g1-g0:.0f}s", "ts": (g0 - T0) * US,
                           "dur": (g1 - g0) * US, "args": {"wall seconds": round(g1 - g0, 1)}})

        # steps this agent's tag owns, matched to its srun launches in order
        # Only start rows carry the agent tag; end rows are keyed by
        # (node, step) alone, so select on starts and pair ends by key.
        mine, starts, steps = set(), {}, []
        for r in ledger:
            k = (r.get("node"), r.get("step"))
            if r["ev"] == "start" and not r.get("miss") and r.get("agent") == tag:
                mine.add(k); starts[k] = r
            elif r["ev"] == "end" and k in starts:
                steps.append((starts.pop(k), r))
        steps.sort(key=lambda x: x[0]["ts"])
        t_step = lane(i, "steps (executed)")
        for n, (st, en) in enumerate(steps):
            gl = st.get("gpus") or []
            ev.append({"ph": "X", "pid": i, "tid": t_step, "cat": "step",
                       "name": f"step {st['step']}  cores {st.get('cpus')}"
                               + (f"  gpu {','.join(str(g['index']) for g in gl)}" if gl else ""),
                       "ts": (st["ts"] - T0) * US, "dur": (en["ts"] - st["ts"]) * US,
                       "args": {"cores": st.get("cpus"),
                                "gpu uuids": [g.get("uuid") for g in gl],
                                "lifetime seconds": en.get("lifetime_s")}})
            # Match launch to step by TIME CONTAINMENT, not by order. srun
            # blocks until its step ends, so the step nests inside the tool call
            # that launched it. Ordering breaks whenever launches are
            # concurrent — `srun A & srun B & wait` in one call, several Bash
            # calls in one turn, or a subagent launching its own — and one call
            # can legitimately own several steps. Among containing launches,
            # the tightest window is the real parent.
            cands = [(lb - la, la) for la, lb, _ in launches
                     if la <= st["ts"] + 0.5 and en["ts"] <= lb + 0.5]
            if cands:
                _, la = min(cands)
                fid = flow_id[0]; flow_id[0] += 1
                ev.append({"ph": "s", "id": fid, "cat": "launch", "name": "srun",
                           "pid": i, "tid": t_tool, "ts": (la - T0) * US})
                ev.append({"ph": "f", "id": fid, "cat": "launch", "name": "srun",
                           "pid": i, "tid": t_step, "ts": (st["ts"] - T0) * US, "bp": "e"})
                unmatched[0] += 0
            else:
                unmatched[0] += 1

    # node-wide counters: how much of the machine is in use, by anyone
    by_ts = {}
    for r in usage:
        b = by_ts.setdefault(round(r["ts"], 0), {"cpu": 0.0, "mem": 0.0})
        if r.get("cpu_delta_pct") is not None:
            b["cpu"] += r["cpu_delta_pct"]
        if r.get("mem_mb") is not None:
            b["mem"] += r["mem_mb"]
    for t, v in sorted(by_ts.items()):
        ev.append({"ph": "C", "pid": 0, "ts": (t - T0) * US,
                   "name": "Node CPU % (all steps)", "args": {"total": round(v["cpu"], 1)}})
        ev.append({"ph": "C", "pid": 0, "ts": (t - T0) * US,
                   "name": "Node memory MB (all steps)", "args": {"total": round(v["mem"], 1)}})

    out = Path(out_path or (runs[0].parent / "combined.trace.json"))
    out.write_text(json.dumps({"traceEvents": ev, "displayTimeUnit": "ms"}))
    print(f"=== combined: {len(agents)} agent(s) ===")
    for tag, (d, calls) in sorted(agents.items()):
        n = sum(1 for e in ev if e["ph"] == "X" and e.get("cat") == "step"
                and e["pid"] == pid[tag])
        print(f"  {tag:<28} {len(calls):>4} tool calls  {n:>3} steps")
    print(f"  flow arrows      {flow_id[0]:>4}"
          + (f"   ({unmatched[0]} step(s) with no containing srun call)" if unmatched[0] else ""))
    print(f"\n  trace: {out}")
    print("  open at https://ui.perfetto.dev")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(args) > 1:
        combine(args)
    else:
        build(args[0] if args else ".")
