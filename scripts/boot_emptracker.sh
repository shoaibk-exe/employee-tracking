#!/usr/bin/env bash
# On machine boot (or any call): start EMPTracker if we are inside the
# weekday operating window. Outside the window, leave services alone.
#
# Window matches cron: Mon–Fri 09:00 inclusive → 18:00 exclusive (PKT).

set -euo pipefail

ROOT="/data/home/emplyee-tracking/custom_sys"
LOG_DIR="$ROOT/logs"
START="$ROOT/scripts/start_emptracker.sh"
CONDA_SH="/home/ai-dw/miniconda3/etc/profile.d/conda.sh"

mkdir -p "$LOG_DIR"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# Wait for conda / start script (disk or env may lag right after boot).
for i in $(seq 1 36); do
  if [[ -x "$START" && -f "$CONDA_SH" ]]; then
    break
  fi
  echo "$(ts) Boot check: waiting for start script/conda (${i}/36)"
  sleep 5
done

if [[ ! -x "$START" ]]; then
  echo "$(ts) Boot check: missing $START — cannot start"
  exit 1
fi
if [[ ! -f "$CONDA_SH" ]]; then
  echo "$(ts) Boot check: missing $CONDA_SH — cannot start"
  exit 1
fi

# %u = 1(Mon)…7(Sun); hhmm as decimal (avoid bash octal on leading zeros).
dow=$(date +%u)
hm=$((10#$(date +%H%M)))

if [[ "$dow" -gt 5 ]]; then
  echo "$(ts) Boot check: weekend (dow=$dow) — not starting"
  exit 0
fi

if [[ "$hm" -lt 900 || "$hm" -ge 1800 ]]; then
  echo "$(ts) Boot check: outside 09:00–18:00 (now=$hm) — not starting"
  exit 0
fi

echo "$(ts) Boot check: weekday window (dow=$dow hm=$hm) — starting EMPTracker"
exec "$START"
