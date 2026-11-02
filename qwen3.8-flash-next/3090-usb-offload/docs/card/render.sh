#!/bin/sh
# Render exl3-card.html at 1:1 (1200x675) with headless Brave into exl3-card.png (same method as local-omarchy/brand/export.sh).
cd "$(dirname "$0")"
B="/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
perl -e 'alarm 60; exec @ARGV' "$B" --headless=new --hide-scrollbars --disable-gpu --force-device-scale-factor=1 \
  --allow-file-access-from-files --virtual-time-budget=6000 --window-size=1200,675 --screenshot="$PWD/exl3-card.png" \
  "file://$PWD/exl3-card.html" >/dev/null 2>&1
ls -la exl3-card.png
