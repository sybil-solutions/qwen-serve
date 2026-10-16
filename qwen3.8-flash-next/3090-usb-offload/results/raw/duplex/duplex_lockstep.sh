#!/bin/zsh
source ~/.chestnut_env
SITE=$(~/tinygrad/.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
export DEV=NV:PTX PYTHONPATH=$HOME/shim:$HOME/tg-strata:$SITE PYTHONNOUSERSITE=1
exec ~/tinygrad/.venv/bin/python ~/shim/duplex_lockstep.py "$@"
