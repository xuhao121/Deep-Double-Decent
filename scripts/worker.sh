#!/usr/bin/env bash
# Run the job list of one lane on one MIG slice, one run after another.
#
#   bash scripts/worker.sh <MIG_UUID> <job_list.txt> [owner]
#
# - Runs with outputs/<run_id>/DONE are skipped; interrupted runs resume from their checkpoint.
# - With [owner], train.py skips runs that runs.csv has meanwhile reassigned to the other member.
# - A run stopped by a signal exits with code >= 128: the worker then stops instead of
#   starting the next run (Ctrl-C in tmux, or scripts/stop_workers.sh, stops the lane cleanly).
set -u
DEV="$1"
LIST="$2"
OWNER="${3:-}"
PY="${PYTHON:-python}"                 # e.g. PYTHON=python3 if there is no `python` on PATH
cd "$(dirname "$0")/.." || exit 1
OUT="${DD_OUTPUTS_DIR:-outputs}"       # same default as src/config.py
mkdir -p logs

while read -r RUN; do
  [ -z "$RUN" ] && continue
  case "$RUN" in \#*) continue ;; esac
  [ -f "$OUT/$RUN/DONE" ] && continue
  echo "[$(date '+%F %T')] start $RUN on $DEV"
  CUDA_VISIBLE_DEVICES="$DEV" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$PY" -m src.train --run "$RUN" ${OWNER:+--owner "$OWNER"} >> "logs/$RUN.log" 2>&1
  RC=$?
  echo "[$(date '+%F %T')] exit $RC $RUN"
  if [ "$RC" -ge 128 ]; then
    echo "run stopped by a signal - worker exits"
    break
  fi
done < "$LIST"
echo "[$(date '+%F %T')] lane finished: $LIST"
