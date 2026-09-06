#!/bin/bash
# Memory watcher: logs free -g every 5s; kills spark serve if free < 6GB.
OUT=/home/ms/.claude/jobs/5a7bd33d/tmp/boot-memwatch.log
while true; do
  line=$(free -g | awk '/^Mem:/ {print strftime("%H:%M:%S"), "total="$2, "used="$3, "free="$4, "avail="$7}')
  echo "$line" >> "$OUT"
  freeg=$(free -g | awk '/^Mem:/ {print $4}')
  if [ "$freeg" -lt 6 ]; then
    echo "$(date +%H:%M:%S) FREE<6GB — killing spark serve" >> "$OUT"
    pkill -f 'target/release/spark serve' >> "$OUT" 2>&1
    exit 1
  fi
  sleep 5
done
