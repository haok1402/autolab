# SLURM-traced AutoLab runs

Three pieces, plus a viewer:

| script | role |
|---|---|
| `run_experiment.sh` | AutoLab's own protocol — one continuous Claude Code session per task, bounded only by `task.toml` |
| `slurm_ledger.py` | records **which** cores and **which** GPU every SLURM step was given |
| `slurm_usage.py` | samples what each step is **using**, for the whole run |
| `visualize.py` | merges all three into a Perfetto timeline |

## Core layout

```
job allocation (e.g. 64 cores, 8 GPUs)
├── monitors     plain processes, pinned to the TOP cores
├── agent        apptainer container, pinned below them
└── experiments  the agent runs `srun --exact -cN ...`
                 -> top-level steps, exclusive cores, recorded by the ledger
```

The agent is deliberately **not** launched with `srun`. A child step of the agent's
own step either pends forever (steps are exclusive by default, so the child waits
on its parent) or needs `--overlap`, which puts the experiment on the agent's own
cores — the agent's thinking would then pollute every benchmark it times.
Running the agent as a plain pinned process keeps experiment steps exclusive.

Monitors sit on the *highest* cores because SLURM allocates step cores from low
upward, so experiments fill from core 0 and do not collide with the infrastructure.

## Getting the allocation (this matters)

Obtain the node with `salloc`, not with an interactive `srun`:

```bash
salloc -N1 -c64 --gres=gpu:8 --mem=0 -t 24:00:00
```

An interactive `srun --pty bash` creates step `.0` that holds the **entire**
allocation (`cpu=64, gres/gpu=8, mem=773000M`). Every later step then either
pends forever (nothing is free) or must pass `--overlap`, in which case
concurrent steps all land on the same cores — measured: four concurrent
`--exact -c2` steps every one of them on cores `0,1`. Exclusive per-experiment
cores, which is the entire point of routing experiments through `srun`, are
only available when no step holds the allocation.

`run_experiment.sh` checks for this and refuses to start rather than producing
timings that silently share cores.

## Usage

```bash
scripts/run_experiment.sh radix_sort          # build, run, score, visualize
scripts/visualize.py $AUTOLAB_WORK/radix_sort # re-render an existing run
```

Outputs land in `$AUTOLAB_WORK/<task>/` (default `/scratch/$USER/autolab/<task>/`):

```
ledger.jsonl    one row per step: agent, cores, GPU uuid, lifetime
usage.jsonl     per-step gauges: cpu%, memory, GPU utilisation
agent.jsonl     Claude Code's stream-json trajectory
result.json     the verifier's reward
profile.trace.json   open at https://ui.perfetto.dev
```

`visualize.py` also drops a copy in `traces/` at the repo root, so the trace is
one click away in an editor file tree. `traces/.gitignore` keeps them out of git.

## Caveats

* `cpuset.cpus.effective` is readable ~6 us after a step's cgroup appears but still
  holds the inherited job-wide set; slurmd narrows it at ~0.85 ms. Never emit on the
  first non-empty read — that is a wrong answer, not an early one.
* Placement is only observable while a step lives. `sacct` records allocation
  **counts** but never which cores or which GPU, so the ledger must catch it live.
* A step's environment appears at ~19 ms (when slurmd execs the task). Steps shorter
  than that cannot be attributed by anyone; they are logged as misses.
* `/dev/nvidiaN` is not `nvidia-smi`'s GPU N. Always key on UUID.
