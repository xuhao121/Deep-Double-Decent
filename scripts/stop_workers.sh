#!/usr/bin/env bash
# Stop all running training processes of this user cleanly: each finishes its current epoch,
# writes a checkpoint and exits; the workers then stop instead of starting new runs.
# Use this before re-assigning jobs (e.g. after adding R3/R6), then re-run assign_jobs.py
# and launch_tmux.sh - interrupted runs resume from their checkpoints.
#
#   bash scripts/stop_workers.sh
pkill -INT -u "$(id -u)" -f "src.train --run" && echo "SIGINT sent; runs checkpoint at the end of their epoch" \
  || echo "no running training processes found"
