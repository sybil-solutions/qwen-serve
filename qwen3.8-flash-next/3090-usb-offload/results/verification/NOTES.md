# Verification notes (2026-10-12)

`verify-card-free-run.txt` is the output of `scripts/verify-card-free.sh` (all six steps) on the Mac mini, run from a clone under `$HOME`, with
`STRATA_DIR` pointing at the mini's Strata git checkout (tag `v0.1.41`) and `LIBEXL3METAL` at a dylib built on the driver Mac. Nothing in it touches the
GPU or the running server: it writes only under `/tmp/repro-work`; the cpu and metal steps read `~/exl3/pack/experts.bin` through a read-only mmap.

Checks done by hand, same day, not part of the script output:

- Driver Mac (Apple silicon, Xcode Metal toolchain): `exl3/metal/build.sh /tmp/metal-build` -> `libexl3metal.dylib` sha256
  `de43149055d826d0cba50f3259d777799d9ac1adb19ded4c80d401d599cbf398`, identical to the dylib deployed on the mini.
- Driver Mac: `git clone --branch v0.1.41 https://github.com/Niko1221/Strata` -> `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`;
  `git apply --check patches/vision/0001-strata-vision-metal-macos.patch` ok; llama.cpp fetched at `3cf03257f219afbe7334045ff7c6a06ac68c627d`;
  `vision/build_vision_mac.sh` built `strata-vision-metal` in 38 s (8,077,952 B, same size as the deployed one; sha256 differs; not run).
- `configs/render.sh <template> /Users/<user>` for the three templates compared with `cmp` against the live config files on the mini: identical.
- Mini: `pack_td.py index` regenerated `td/experts_index.json` / `other_index.json` byte-identical to the existing files (sha256 `2dc6b0ce77e9...`,
  `0ea6ec8cdcc5...`). The first run of `td_hdr.py` (which then wrote to `~/exl3/td/`) rewrote the live copies of those two files with identical bytes;
  the script now writes next to itself.
- Mini: `python3 -c "px.marker_text(manifest) == exl3/pack_full512/exl3_experts.txt"` true (marker regenerated from the manifest).
- `exl3/make_profile512_freq.py` regenerates `exl3/placement/f512c/profile.bin` byte for byte from `exl3/placement/inputs/` + Strata's `tools/make_profile.py`
  (sha256 `6e3653c1ea9a...`).
- `docker build` of `shim/docker/Dockerfile.strata-cuda` under a throw-away tag succeeded (cached layers); the tag was removed, `strata-cuda:latest` untouched.
- fatbin / libcudart build-flag experiments: see section 8 of `docs/results.md`.

Not done: a full Strata engine rebuild (CMake configure + `macbuild.py` + ninja), any run that needs the card, any request to the live server.
