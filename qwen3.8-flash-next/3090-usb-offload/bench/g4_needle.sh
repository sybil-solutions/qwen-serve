#!/bin/zsh
# g4_needle.sh NAME CFG [TOKENS] : start CFG, run the needle at TOKENS (default 32000) -> ~/shim/g4_NAME.log
name=$1; cfg=$2; tok=${3:-32000}
out=~/shim/g4_$name.log
{
echo "== $name $(date -u +%FT%TZ) cfg=$cfg"
~/shim/ls_srv.sh $cfg || { echo "SERVER FAILED"; exit 1; }
echo "-- needle $tok"
python3 ~/shim/final_long.py needle $tok 50
echo "== done $(date -u +%FT%TZ)"
} > $out 2>&1
