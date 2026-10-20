#!/bin/bash
# Board launcher: run_session as a daemon + a memory watchdog.
# The board hung and rebooted at MemAvailable ~85 MB on 2026-10-20 (4 resident servers), so the watchdog kills
# every exp2.serve process when MemAvailable drops below LIMIT_MB.
# usage: launch.sh PLAN OUTDIR NAME [LIMIT_MB] [TYPES]
# TYPES defaults to the synthetic example; Run 2 must pass the frozen types file (run_session exits 3 on a hash mismatch).
set -e
PLAN=$1; OUT=$2; NAME=$3; LIMIT=${4:-250}   # hang seen at 85-150 MB; 3-family smoke ran >= 509 MB, 352 MB at teardown
TYPES=${5:-exp2/configs/example_types.json}
BASE=$(dirname "$OUT"); PY=/home/earlyexp/EarlyExitMonoRepo/venv/bin/python
cd ~/exp2_wt
rm -rf "$OUT"
PYTHONDONTWRITEBYTECODE=1 JETSON_SUDO_PW=nopasswd-dummy setsid nohup "$PY" -m exp2.serve.run_session \
    --plan "$PLAN" --types "$TYPES" --out "$OUT" >> "$BASE/$NAME.log" 2>&1 < /dev/null &
echo $! > "$BASE/$NAME.pid"
setsid nohup bash -c '
  pid=$(cat "$1"); limit=$2; log=$3
  while kill -0 "$pid" 2>/dev/null; do
    m=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo)
    if [ "$m" -lt "$limit" ]; then
      echo "$(date +%T) watchdog: MemAvailable ${m} MB < ${limit} MB, killing exp2.serve" >> "$log"
      pkill -TERM -f "exp2.serve"; sleep 5; pkill -KILL -f "exp2.serve"; exit 1
    fi
    sleep 1
  done' _ "$BASE/$NAME.pid" "$LIMIT" "$BASE/$NAME.log" > /dev/null 2>&1 < /dev/null &
echo "PID=$(cat "$BASE/$NAME.pid") watchdog=$! limit=${LIMIT}MB log=$BASE/$NAME.log"
