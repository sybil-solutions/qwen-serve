#!/bin/zsh
# final_g4.sh : restart on the final G=4 config, 32K needle, vision, 190K needle -> ~/shim/final_g4.log
{
echo "== final_g4 $(date -u +%FT%TZ)"
~/shim/ls_srv.sh ~/shim/strata-exl3-full512-final-g4.json || { echo "SERVER FAILED"; exit 1; }
echo "-- needle 32000"; python3 ~/shim/final_long.py needle 32000 50
echo "-- vision"; python3 ~/shim/final_vision.py
echo "-- needle 190000"; python3 ~/shim/final_long.py needle 190000 50
echo "== done $(date -u +%FT%TZ)"
} > ~/shim/final_g4.log 2>&1
