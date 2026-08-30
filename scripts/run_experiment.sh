#!/usr/bin/env bash
# run_experiment.sh — AutoLab's own protocol, traced through SLURM.
#
# One continuous Claude Code session per task, bounded only by what task.toml
# declares. No turn cap, no rounds, no intermediate feedback: the agent runs
# until it stops or the clock expires, and is scored once at the end.
#
#   scripts/run_experiment.sh <task> [--dry-run]
#
# Shape, inside the current SLURM allocation:
#
#   monitors      exclusive 1-core steps (ledger + usage)
#   agent         an exclusive step, running INSIDE the task image
#   experiments   the agent calls srun itself, choosing its own resources;
#                 each step re-enters the task image via apptainer
#
# An srun launched from inside a step inherits that step's CPU mask, so a nested
# --exact would fail with "CPU binding outside of job step allocation". Setting
# SLURM_CPU_BIND=none in the agent's environment lifts that inheritance, so the
# agent can simply call srun and its experiments get their own exclusive cores
# (measured: two concurrent nested steps landed on 9,10 and 11,12 while the
# agent held 1-8). The step's own cpuset still binds, so isolation is real.
set -euo pipefail

# Bash reads a script incrementally by byte offset, so editing this file while a
# run is in flight corrupts the running shell. Re-exec from a private snapshot.
if [ -z "${AUTOLAB_PINNED:-}" ]; then
    AUTOLAB_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export AUTOLAB_REPO
    _snap="$(mktemp /tmp/run_experiment.XXXXXX.sh)"; cat "${BASH_SOURCE[0]}" > "$_snap"
    chmod +x "$_snap"; export AUTOLAB_PINNED="$_snap"; exec "$_snap" "$@"
fi

REPO="${AUTOLAB_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TASK="${1:?usage: run_experiment.sh <task> [--dry-run]}"
DRY=""; [ "${2:-}" = "--dry-run" ] && DRY=1
TASK_DIR="$REPO/tasks/$TASK"
[ -d "$TASK_DIR" ] || { echo "no such task: $TASK" >&2; exit 1; }

# Concurrent runs of the SAME task must not share /app: a bind mount is one
# host directory seen from several containers, with no copy-on-write, so two
# agents would edit the same solve.c and the verifier would score whichever
# write landed last. Set AUTOLAB_RUN_ID to give a run its own tree; the built
# image and the staged SLURM libraries are read-only and stay shared.
BASE="${AUTOLAB_WORK:-/scratch/$USER/autolab}/$TASK"
RUN_ID="${AUTOLAB_RUN_ID:-}"
WORK="$BASE${RUN_ID:+-$RUN_ID}"
SIF="$BASE/env.sif"; SLURMLIBS="$BASE/slurmlibs"
APP="$WORK/app"; OUT="$WORK"
CC_SRC="$HOME/.local/share/claude"
JOB="${SLURM_JOB_ID:?must run inside a SLURM allocation}"

toml() { grep -E "^\s*$1\s*=" "$TASK_DIR/task.toml" | head -1 | sed 's/.*=\s*//;s/\s*#.*//' | tr -d '"'; }
AGENT_TIMEOUT="${AUTOLAB_TIMEOUT_OVERRIDE:-$(toml timeout_sec)}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-7200}"
VERIFIER_TIMEOUT="$(grep -A3 '^\[verifier\]' "$TASK_DIR/task.toml" | grep timeout_sec | head -1 |
                    sed 's/.*=\s*//;s/\s*#.*//')"; VERIFIER_TIMEOUT="${VERIFIER_TIMEOUT:-300}"
TASK_CPUS="$(toml cpus)"; TASK_CPUS="${TASK_CPUS:-2}"
AGENT_NCPU="${AUTOLAB_AGENT_CPUS:-8}"

# A step holding the whole allocation makes exclusive experiment steps
# impossible: without --overlap they pend forever, and with it they all land on
# the same cores. Refuse rather than emit timings that quietly share CPUs.
# Usually caused by a bare `srun` in the batch script (it inherits everything);
# give it --exact --cpus-per-task=1 --mem=<small> --gres=none.
HOG=$(scontrol -d show step "$JOB" 2>/dev/null |
      awk -v n="$(nproc)" '/^StepId=/{id=$1} /TRES=cpu=/{ if ($0 ~ "cpu="n",") print id }' | head -1)
if [ -n "$HOG" ] && [ -z "${AUTOLAB_ALLOW_SHARED_CORES:-}" ]; then
    echo "ERROR: $HOG holds all $(nproc) CPUs of this allocation; experiment steps" >&2
    echo "       cannot get exclusive cores. Constrain the step that holds them" >&2
    echo "       (--exact --cpus-per-task=1 --mem=32M --gres=none), or set" >&2
    echo "       AUTOLAB_ALLOW_SHARED_CORES=1 to proceed with unsound timings." >&2
    exit 1
fi

echo "task            $TASK${RUN_ID:+  [run $RUN_ID]}   (job $JOB on $(hostname))"
echo "agent budget    ${AGENT_TIMEOUT}s     verifier ${VERIFIER_TIMEOUT}s"
echo "agent           $AGENT_NCPU-core exclusive step"
echo "monitors        1 exclusive core each"
echo "experiments     agent-driven srun steps (it picks resources)"
[ -n "$DRY" ] && { echo "(dry run — stopping here)"; exit 0; }

mkdir -p "$OUT"

# ── task image ──────────────────────────────────────────────────────────────
mkdir -p "$BASE"
# mkdir is atomic, so it serialises concurrent first-runs of the same task
while ! mkdir "$BASE/.build.lock" 2>/dev/null; do
    [ -f "$SIF" ] && break
    sleep 5
done
if [ ! -f "$SIF" ]; then
    echo ">> building task image"
    python3 "$REPO/scripts/dockerfile2def.py" "$TASK_DIR/environment/Dockerfile" > "$BASE/env.def"
    ( cd "$TASK_DIR/environment" && apptainer build --force "$SIF" "$BASE/env.def" )
fi
rm -rf "$APP"; mkdir -p "$APP"
apptainer exec "$SIF" cp -a /app/. "$APP/"

# ── SLURM client bits the task image lacks ──────────────────────────────────
# This site runs auth/slurm (not munge) with configless mode. A hand-written
# library list is wrong for any image but the one it was written against, so
# compute the closure: everything srun and the plugins link, minus what the
# image already has. srun also refuses to start without a `slurm` user.
if [ ! -f "$SLURMLIBS/.complete" ]; then
    echo ">> staging SLURM client libraries"
    mkdir -p "$SLURMLIBS"
    { ldd /usr/bin/srun; for f in /usr/lib64/slurm/*.so; do ldd "$f" 2>/dev/null || true; done; } \
        | awk '/=>/ && $3 ~ /^\// {print $1" "$3}' | sort -u > "$BASE/.deps"
    apptainer exec "$SIF" bash -c 'ldconfig -p | awk "{print \$1}"' 2>/dev/null | sort -u > "$BASE/.have"
    n=0
    while read -r name path; do
        if ! grep -qx "$name" "$BASE/.have"; then
            cp -L "$path" "$SLURMLIBS/" 2>/dev/null && n=$((n+1))
        fi
    done < "$BASE/.deps"
    cp -L /usr/lib64/slurm/libslurmfull.so "$SLURMLIBS/" 2>/dev/null || true
    rm -f "$BASE/.deps" "$BASE/.have"; touch "$SLURMLIBS/.complete"
    echo "   staged $n libraries"
fi
rmdir "$BASE/.build.lock" 2>/dev/null || true
apptainer exec "$SIF" cat /etc/passwd > "$OUT/passwd"
apptainer exec "$SIF" cat /etc/group  > "$OUT/group"
getent passwd slurm >> "$OUT/passwd"; getent passwd "$USER" >> "$OUT/passwd"
getent group  slurm >> "$OUT/group";  getent group "$(id -gn)" >> "$OUT/group"

# ── agent credentials ───────────────────────────────────────────────────────
# Refresh every run, never "only if missing": OAuth tokens rotate, and a stale
# copy fails with "OAuth session expired" while the host's own are valid.
HOMEDIR="$WORK/agenthome"; mkdir -p "$HOMEDIR/.claude"
chmod 700 "$WORK" "$HOMEDIR" "$HOMEDIR/.claude"
[ -f "$HOME/.claude/.credentials.json" ] &&
    install -m 600 "$HOME/.claude/.credentials.json" "$HOMEDIR/.claude/.credentials.json"
CC_BIN="versions/$(ls -1 "$CC_SRC/versions" | sort -V | tail -1)"

# ── observers, one set per JOB ──────────────────────────────────────────────
# The ledger and sampler watch the whole job's cgroup tree, so they see every
# step from every agent. Running a pair per agent would burn 2N cores to write
# N identical files. Start them once; whoever gets there first owns them, and
# they are deliberately NOT killed at the end of a run — later runs in the same
# job keep using them, and they exit when the job does.
OBS="${AUTOLAB_WORK:-/scratch/$USER/autolab}/observers-$JOB"
mkdir -p "$OBS"
LEDGER=""; USAGE=""
if mkdir "$OBS/.owner" 2>/dev/null; then
    echo ">> starting observers for job $JOB (shared by every run in this job)"
    setsid srun --exact -n1 -c1 --mem=1G --gres=none --job-name=obs-ledger \
        python3 "$REPO/scripts/slurm_ledger.py" --out "$OBS/ledger.jsonl" --job "$JOB" \
        >/dev/null 2>"$OBS/ledger.err" < /dev/null &
    LEDGER=$!
    setsid srun --exact -n1 -c1 --mem=1G --gres=none --job-name=obs-usage \
        python3 "$REPO/scripts/slurm_usage.py" --out "$OBS/usage.jsonl" --job "$JOB" --interval 1 \
        >/dev/null 2>"$OBS/usage.err" < /dev/null &
    USAGE=$!
    sleep 3
else
    echo ">> observers already running for job $JOB ($OBS)"
fi
trap 'rm -f "${AUTOLAB_PINNED:-}"' EXIT

# ── the agent's brief ───────────────────────────────────────────────────────
PROMPT="$OUT/prompt.md"
cp "$TASK_DIR/instruction.md" "$PROMPT"
{
  echo
  echo "## Running experiments"
  echo
  echo "You are on a SLURM compute node, inside this task's container image."
  echo "Run every build, benchmark and measurement as a SLURM step, never"
  echo "directly in this session:"
  echo
  echo "    srun -c${TASK_CPUS} apptainer exec \$AUTOLAB_SIF bash -lc '<your command>'"
  echo
  echo "where <your command> is whatever the task above describes for building"
  echo "and running — a make invocation, a cargo build, a python script."
  echo
  echo "The apptainer part re-enters this same image, so the step has the same"
  echo "compilers and tools you see here. Everything else — the job id, the"
  echo "working directory, the bind mounts — is already set in the environment."
  echo
  echo "You choose the resources: -c<N> for cores, --mem=<M>, --gres=gpu:<N>."
  echo "This task declares ${TASK_CPUS} CPUs; more will not speed up a"
  echo "single-threaded benchmark, and larger requests wait longer for a slot."
  echo "A step you do not constrain inherits what the job holds, so ask for what"
  echo "the work needs."
  echo
  echo "Why this is about correctness, not bookkeeping: a step gets its own"
  echo "exclusive CPU cores, so its timings are not polluted by this agent"
  echo "process or by anything else on the node. A command run directly here"
  echo "competes with this session for CPU and its numbers cannot be trusted."
  echo
  echo "Reading and editing files needs none of this — only things you intend to"
  echo "time. A step may wait briefly when the node is busy; that is normal."
} >> "$PROMPT"

# ── the session ─────────────────────────────────────────────────────────────
echo ">> agent starting ($AGENT_NCPU-core step, budget ${AGENT_TIMEOUT}s)"
date +%s > "$OUT/t0.epoch"
RC=0
timeout --signal=TERM --kill-after=60 "$AGENT_TIMEOUT" \
srun --exact -n1 -c"$AGENT_NCPU" --mem=16G --gres=none --job-name=agent \
apptainer exec --cleanenv --writable-tmpfs \
    --bind "$CC_SRC:/mnt:ro" --bind "$HOMEDIR:/agenthome" --bind "$APP:/app" \
    --bind "$OUT/passwd:/etc/passwd" --bind "$OUT/group:/etc/group" \
    --bind /usr/bin/srun --bind /usr/bin/squeue --bind /usr/lib64/slurm --bind /etc/slurm \
    --bind /run/slurm --bind /var/spool/slurmd/conf-cache \
    --bind "$SLURMLIBS:$SLURMLIBS" --pwd /app \
    --env CLAUDE_CONFIG_DIR=/agenthome/.claude --env IS_SANDBOX=1 \
    --env SLURM_CONF_SERVER="${AUTOLAB_CONF_SERVER:-controller0}" \
    --env LD_LIBRARY_PATH="$SLURMLIBS" --env SLURM_JOB_ID="$JOB" \
    --env SLURM_CPU_BIND=none \
    --env SLURM_EXACT=1 --env SLURM_NTASKS=1 --env SLURM_WORKING_DIR="$WORK" \
    --env APPTAINER_BIND="$APP:/app" --env APPTAINER_PWD=/app \
    --env AUTOLAB_SIF="$SIF" \
    --env AGENT_ID="${AUTOLAB_AGENT_ID:-agent-$TASK${RUN_ID:+-$RUN_ID}}" \
    --env CLAUDE_CODE_OAUTH_TOKEN="${CLAUDE_CODE_OAUTH_TOKEN:-}" \
    --env ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
    "$SIF" "/mnt/$CC_BIN" --print --verbose --output-format=stream-json \
        --permission-mode=bypassPermissions -- "$(cat "$PROMPT")" \
    > "$OUT/agent.jsonl" 2>&1 || RC=$?

ELAPSED=$(( $(date +%s) - $(cat "$OUT/t0.epoch") ))
case "$RC" in
  124|137) echo ">> agent hit the ${AGENT_TIMEOUT}s budget after ${ELAPSED}s" ;;
  0)       echo ">> agent finished on its own after ${ELAPSED}s" ;;
  *)       echo ">> agent exited rc=$RC after ${ELAPSED}s" ;;
esac

# ── score whatever it left behind ───────────────────────────────────────────
echo ">> scoring"
mkdir -p "$OUT/verifier"
timeout --signal=TERM --kill-after=30 "$VERIFIER_TIMEOUT" \
apptainer exec --cleanenv --bind "$APP:/app" --bind "$OUT:/logs" \
    --bind "$TASK_DIR/tests:/tests" --pwd /app "$SIF" \
    bash -c 'mkdir -p /logs/verifier && bash /tests/test.sh' > "$OUT/verify.log" 2>&1 || true
cp "$OUT/verifier/reward.json" "$OUT/result.json" 2>/dev/null || echo '{"reward":null}' > "$OUT/result.json"
echo "   $(cat "$OUT/result.json")"

# Observers stay up for other runs in this job. Snapshot what they have
# recorded so this run's trace is self-contained.
cp "$OBS/ledger.jsonl" "$OUT/ledger.jsonl" 2>/dev/null || true
cp "$OBS/usage.jsonl"  "$OUT/usage.jsonl"  2>/dev/null || true
python3 "$REPO/scripts/visualize.py" "$OUT" || true
echo ">> done: $OUT"
