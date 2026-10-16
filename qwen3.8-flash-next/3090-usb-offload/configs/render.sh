#!/bin/sh
# render.sh <config.json.tmpl> [HOME_DIR] > config.json : fill the {{HOME}} placeholder (default $HOME). The server reads the result with
#   cd ~/strata && STRATA_PREFILL_CPU_SHARE=1 .venv/bin/python serve/server.py --engine strata --config <config.json> --port 8080
sed "s#{{HOME}}#${2:-$HOME}#g" "$1"
