#!/usr/bin/env bash
# Start every lane listed in lanes.tsv (written by assign_jobs.py) in its own tmux window.
#
#   bash scripts/launch_tmux.sh jobs/m1/lanes.tsv m1
#
# Attach with `tmux attach -t dd-m1`; switch windows with Ctrl-b n / Ctrl-b p; detach with Ctrl-b d.
set -eu
LANES="$1"
OWNER="$2"
SESSION="dd-$OWNER"
cd "$(dirname "$0")/.."

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION" -n monitor
  tmux send-keys -t "$SESSION:monitor" "watch -n 300 python scripts/status.py --owner $OWNER" C-m
fi
tail -n +2 "$LANES" | while IFS=$'\t' read -r LANE DEV LIST; do
  tmux new-window -t "$SESSION" -n "$LANE" "bash scripts/worker.sh '$DEV' '$LIST' '$OWNER'; echo 'worker ended'; bash"
  echo "started $LANE on $DEV ($(wc -l < "$LIST") runs)"
done
echo "tmux session: $SESSION   (tmux attach -t $SESSION)"
