#!/bin/zsh
# libstrata_exl3.dylib: host stubs + C API embedding ../exl3_strata.fatbin, ../exl3_strata_recon.fatbin and ../exl3_strata_pf.fatbin, linked to the
# tgcu libcudart (the same shim instance Strata uses, so the fatbins register with it).  Run ../build.sh exl3_strata and
# ../build.sh exl3_strata_recon and ../build.sh exl3_strata_pf first.
set -e
cd ${0:A:h}
SDK=${SDK:-/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk}
host() {  # host() <src> <fatbin> <obj>
  clang++ -x cuda --cuda-host-only --no-offload-new-driver --cuda-path=$HOME/cuda-12.8 -nocudalib -include cuda_runtime.h \
    -include $HOME/shim/tgcuda_launch.h -Xclang -fcuda-include-gpubinary -Xclang $2 -std=c++17 -O2 -fPIC -isystem $HOME/shim/include \
    -isystem $HOME/cuda-12.8/include -I ../torchstub -isysroot $SDK -Wno-unknown-cuda-version -c $1 -o $3
}
host exl3_strata_host.cu ../exl3_strata.fatbin exl3_strata_host.o
host exl3_strata_recon_host.cu ../exl3_strata_recon.fatbin exl3_strata_recon_host.o
host exl3_strata_pf_host.cu ../exl3_strata_pf.fatbin exl3_strata_pf_host.o
clang++ -shared -O2 exl3_strata_host.o exl3_strata_recon_host.o exl3_strata_pf_host.o -o libstrata_exl3.dylib -L$HOME/shim -lcudart \
  -Wl,-rpath,$HOME/shim -isysroot $SDK
rm -f *.o
nm -gU libstrata_exl3.dylib | grep exl3s_
shasum -a 256 libstrata_exl3.dylib
