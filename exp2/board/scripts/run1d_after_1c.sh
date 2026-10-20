#!/bin/bash
# Start Run 1d (LLaMA co-run cells, longer windows) as soon as Run 1c has finished (2026-10-20).
B=/home/earlyexp/exp2_pilot_20261020
until grep -q "all done" $B/run1c.log 2>/dev/null; do sleep 20; done
sleep 60
bash /home/earlyexp/exp2_wt/exp2/board/scripts/launch.sh $B/plan_run1d_llama.json $B/session_run1d run1d 250
echo "$(date +%T) run1d launched" >> $B/run1c.log
