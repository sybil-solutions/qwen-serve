#!/usr/bin/env python3
# Turn Strata's Linux CUDA build (ninja -t commands strata) into two builds:
#   device.ninja (runs in the Linux CUDA container): every .cu -> sm_86 fatbin with the same flags
#   host.ninja   (runs on macOS): .cu -> clang --cuda-host-only object embedding its fatbin, .c/.cpp -> clang, link against tgcu libcudart
import os, re, shlex, sys
from pathlib import Path

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--lin", default=str(Path.home() / "strata-linux"))   # the Linux configure ($HOME is mounted 1:1 in the container)
ap.add_argument("--cmds", default=None)                               # default: <lin>/strata_cmds.txt
ap.add_argument("--targets", default="strata")                         # executables to link, space separated
ap.add_argument("--suffix", default="")                                # device<suffix>.ninja / host<suffix>.ninja
args = ap.parse_args()
HOME = Path.home()
SRC = HOME / "strata"
LIN = Path(args.lin)
OUT = HOME / "strata-mac"
TARGETS = args.targets.split()
SHIM = HOME / "shim"
CUDA_HOST = HOME / "cuda-12.8"       # headers clang understands
CUDA_DEV = HOME / "cuda-13.4"
SDK = os.environ.get("SDK", "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk")   # run.sh picks the newest installed 26.x SDK (the 27.0 SDK breaks ld)
X86 = re.compile(r"^-m(avx\w*|sse\w*|f16c|fma|bmi\w*|lzcnt|popcnt|pclmul|aes|vpclmulqdq|gfni|amx\w*|arch=.*|tune=.*)$")

cmds = Path(args.cmds or LIN / "strata_cmds.txt").read_text().splitlines()
dev_rules, host_rules, objs_by_lib, final_links = [], [], {}, {}

def outpath(p:str) -> str: return str(OUT / p) if not p.startswith("/") else p.replace(str(LIN), str(OUT))
def objpath(p:str) -> str: return str(OUT / p) if not p.startswith("/") else p

def flags_of(argv:list[str], keep_x86=False):
  # the defines, include paths, standard and a few codegen flags of one compile command
  out, i = [], 0
  while i < len(argv):
    a = argv[i]
    if a in ("-MD", "-lineinfo") or a.startswith(("--", "-forward-", "-Xfatbin", "-Xptxas")): i += 1; continue  # nvcc-only
    if a in ("-Xcompiler", "-Xptxas", "-Xnvlink", "-Xfatbin", "-maxrregcount"): i += 2; continue
    if a in ("-MT", "-MF", "-o", "-c", "-x"): i += 2; continue
    if a.startswith(("-D_XOPEN_SOURCE", "-D_POSIX_C_SOURCE", "-D_GNU_SOURCE")): pass
    elif a.startswith(("-D", "-I", "-std=", "-O", "-W", "-f", "-g")) and not a.startswith("-Werror"): out.append(a)
    elif a in ("-isystem", "-include", "-iquote"): out += [a, argv[i + 1]]; i += 1
    elif a.startswith("-m") and (keep_x86 or not X86.match(a)): out.append(a)
    i += 1
  return out

for line in cmds:
  for part in [p.strip() for p in line.split("&&")]:
    if not part or part == ":": continue
    argv = shlex.split(part)
    exe = os.path.basename(argv[0])
    if exe == "nvcc" and "-c" in argv:
      src, obj = argv[argv.index("-c") + 1], argv[argv.index("-o") + 1]
      fatbin = outpath(obj) + ".fatbin"
      # device side: same flags, sm_86 SASS only, uncompressed so tgcu can read the cubins
      dflags = [a for a in argv[1:] if not a.startswith("--generate-code") and a not in ("-MD",)]
      j, cleaned = 0, []
      while j < len(dflags):
        if dflags[j] in ("-MT", "-MF", "-o", "-c"): j += 2; continue
        if dflags[j] == "-x": j += 2; continue
        cleaned.append(dflags[j]); j += 1
      dev_rules.append((fatbin, src, cleaned))
      host_rules.append(("cu", outpath(obj), src, flags_of(argv[1:]), fatbin))
    elif exe in ("c++", "g++", "cc", "gcc") and "-c" in argv:
      src, obj = argv[argv.index("-c") + 1], argv[argv.index("-o") + 1]
      host_rules.append(("cxx" if exe in ("c++", "g++") else "c", outpath(obj), src, flags_of(argv[1:]), None))
    elif exe == "ar":
      lib, objs = argv[2], argv[3:]
      objs_by_lib[outpath(lib)] = [outpath(o) for o in objs]
    elif exe in ("c++", "g++") and "-o" in argv and argv[argv.index("-o") + 1] in TARGETS:
      final_links[argv[argv.index("-o") + 1]] = argv

def q(s:str) -> str: return s.replace("$", "$$").replace(" ", "$ ").replace(":", "$:")
def j(xs) -> str: return " ".join(shlex.quote(x).replace("$", "$$") for x in xs)

OUT.mkdir(parents=True, exist_ok=True)
# *** device.ninja (Linux container) ***
with open(OUT / f"device{args.suffix}.ninja", "w") as f:
  f.write(f"rule nvcc\n  command = {CUDA_DEV}/bin/nvcc $flags --fatbin -arch=sm_86 -no-compress -x cu $in -o $out\n  description = NVCC $out\n")
  for fatbin, src, fl in dev_rules:
    fl = [a for a in fl if not a.startswith("-forward-unknown")] + ["-I" + str(SHIM / "include")]
    f.write(f"build {q(fatbin)}: nvcc {q(src)}\n  flags = {j(fl)}\n")

# *** host.ninja (macOS) ***
common = ["-isysroot", SDK, "-mcpu=apple-m4", "-DSTRATA_TGCU=1", "-D_DARWIN_C_SOURCE", "-Wno-unknown-cuda-version", "-Wno-everything"]
def fix_inc(fl):
  out = []
  for a in fl:
    a = a.replace(str(CUDA_DEV / "include"), str(CUDA_HOST / "include"))
    out.append(a)
  return out + ["-isystem", str(SHIM / "include"), "-isystem", str(CUDA_HOST / "include")]

with open(OUT / f"host{args.suffix}.ninja", "w") as f:
  f.write("rule cuhost\n  command = clang++ -x cuda --cuda-host-only --no-offload-new-driver --cuda-path=" + str(CUDA_HOST) +
          " -nocudalib -include cuda_runtime.h -include " + str(SHIM / "tgcuda_launch.h") +
          " -Xclang -fcuda-include-gpubinary -Xclang $fatbin $flags -c $in -o $out\n  description = CUHOST $out\n")
  f.write("rule cxx\n  command = clang++ -x c++ $flags -c $in -o $out\n  description = CXX $out\n")
  f.write("rule cc\n  command = clang -x c $flags -c $in -o $out\n  description = CC $out\n")
  f.write("rule ar\n  command = rm -f $out && libtool -static -o $out $in\n  description = AR $out\n")
  f.write("rule link\n  command = clang++ $flags $in -o $out $libs\n  description = LINK $out\n")
  for kind, obj, src, fl, fatbin in host_rules:
    fl = [a for a in fix_inc(fl) if not a.startswith("-std=c++") or kind != "c"] + common
    if kind == "cu":
      if "_deps/" not in src: fl = ["-include", str(SRC / "include/strata/platform/macos_compat.hpp")] + fl
      f.write(f"build {q(obj)}: cuhost {q(src)} | {q(fatbin)}\n  fatbin = {q(fatbin)}\n  flags = {j(fl)}\n")
    else:
      if "_deps/" not in src:  # Strata's own sources: x86 kernels through SIMDe, Linux calls through macos_compat
        fl = ["-isystem", str(SHIM / "x86compat"), "-isystem", str(HOME / "simde"), "-include", str(SRC / "include/strata/platform/macos_compat.hpp")] + fl
      f.write(f"build {q(obj)}: {'cxx' if kind == 'cxx' else 'cc'} {q(src)}\n  flags = {j(fl)}\n")
  for lib, objs in objs_by_lib.items(): f.write(f"build {q(lib)}: ar {' '.join(q(o) for o in objs)}\n")
  assert final_links, f"no link command for {TARGETS}"
  for target, final_link in final_links.items():
    ins, libs = [], []
    for a in final_link[1:]:
      if a.endswith(".o"): ins.append(outpath(a))
      elif a.endswith(".a") and not os.path.basename(a).startswith(("libcudart", "libcudadevrt")):
        p = outpath(a)
        if p in objs_by_lib: libs.append(p)
    libs = list(dict.fromkeys(libs))
    link_libs = libs + libs + [f"-L{SHIM}", "-lcudart", f"-Wl,-rpath,{SHIM}", "-isysroot", SDK]
    f.write(f"build {q(str(OUT / target))}: link {' '.join(q(o) for o in ins)} | {' '.join(q(l) for l in libs)}\n"
            f"  flags = -O2\n  libs = {j(link_libs)}\n")
  f.write("default " + " ".join(q(str(OUT / t)) for t in final_links) + "\n")
print(f"device rules {len(dev_rules)}, host rules {len(host_rules)}, libs {len(objs_by_lib)}, links {sorted(final_links)}")
