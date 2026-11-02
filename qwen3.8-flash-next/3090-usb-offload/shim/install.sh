#!/bin/zsh
# Install the shim into $DEST (default $HOME/shim: where run-strata.sh, macbuild.py, the engine's rpath and tgcu_api.py's relative file lookups expect it).
# Copies sources + prebuilt blobs, builds libcudart.dylib.  Overwrites; do not point it at a shim that is serving.
set -e
cd ${0:A:h}
DEST=${1:-$HOME/shim}; mkdir -p $DEST
cp tgcu.py tgcu_api.py tgcu_prof.py usb_drain.py pcie_cycle.py tgcudart.cpp tgcublas.cpp tgcuda_launch.h tgcu_builtin.cu tgcu_libc.ptx \
   run-strata.sh srv.sh ls_srv.sh mkcfg.py macbuild.py build-libcudart.sh $DEST/
cp prebuilt/tgcu_builtin.fatbin prebuilt/tgcu_libc.cubin $DEST/
cp -R include x86compat $DEST/
chmod +x $DEST/*.sh
zsh build-libcudart.sh $DEST
