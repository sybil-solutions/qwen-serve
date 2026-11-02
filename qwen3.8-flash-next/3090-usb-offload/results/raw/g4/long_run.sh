#!/bin/zsh
# long_run.sh NAME CFG TOKENS DEPTH : restart on CFG, needle at TOKENS -> ~/shim/lr_NAME.log
name=$1; cfg=$2; tok=$3; dep=$4
{
echo "== $name $(date -u +%FT%TZ) cfg=$cfg"
~/shim/ls_srv.sh $cfg || { echo "SERVER FAILED"; exit 1; }
echo "-- needle $tok"; python3 ~/shim/final_long.py needle $tok $dep
echo "== done $(date -u +%FT%TZ)"
} > ~/shim/lr_$name.log 2>&1
