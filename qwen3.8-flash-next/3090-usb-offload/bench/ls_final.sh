#!/bin/zsh
{ ~/shim/srv.sh ~/shim/strata-exl3-full512-final.json; python3 ~/shim/ls_pf.py 10000 25000 55000; python3 ~/shim/ls_exact.py LS_final; python3 ~/shim/split_probe.py LS_final 300; echo done; } > ~/shim/ls_final.log 2>&1
