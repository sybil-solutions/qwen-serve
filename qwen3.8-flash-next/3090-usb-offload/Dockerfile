# syntax=docker/dockerfile:1.7
#
# Reproducible build of everything that can be built off the Mac mini: the CUDA device code (sm_86 fatbins of the EXL3 kernels, the shim's two blobs and
# every .cu of the Strata engine), Strata v0.1.41 with the patches applied, and the CMake configure tree the Mac host build is generated from.
# The serving runtime itself is native macOS (libusb to the dock, Metal) and does not run in a container: `./run.sh` consumes this bundle.
#
#   docker compose run --rm build            # or: make        -> ./out  (bundle)
#   docker compose --profile fetch run --rm fetch all          # ~50 GB: experts (sha256-verified) + dense GGUF/mmproj, behind an explicit argument
#
# Everything is pinned: base image by digest, apt packages by version, the CUDA toolkit by sha256 of each redist tarball, Strata by commit.
# Platform is linux/arm64 (the redist tarballs are linux-sbsa; on an Apple-silicon Mac that is native, elsewhere it is emulated and slow).
# apt: only the packages named below are version-pinned; their dependencies come from the same archive snapshot of the day and float.
# The Ubuntu archive drops superseded versions: if a pin is gone the build stops at apt with "Version ... not found" (then update the pins).

ARG UBUNTU=ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55
# The mini's $HOME. exl3_strata.fatbin embeds a build-path-dependent id, so the three source trees are built at exactly the path the shipped one
# was built at (results/verification/fatbin-sha256.txt); the other fatbins do not depend on it.
ARG BUILD_ROOT=/Users/sero

# ---------------------------------------------------------------- toolchain: apt tools + nvcc 13.4.92
FROM --platform=linux/arm64 ${UBUNTU} AS toolchain
ARG BUILD_ROOT
ENV DEBIAN_FRONTEND=noninteractive BUILD_ROOT=${BUILD_ROOT} HOME=${BUILD_ROOT} LC_ALL=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
      g++=4:13.2.0-7ubuntu1 make=4.3-4.1build2 cmake=3.28.3-1build7 ninja-build=1.11.1-2 python3=3.12.3-0ubuntu2.1 \
      xz-utils=5.6.1+really5.4.5-1ubuntu0.3 file=1:5.45-3build1 git=1:2.43.0-1ubuntu7.3 ca-certificates=20260601~24.04.1 curl=8.5.0-2ubuntu10.15 \
 && rm -rf /var/lib/apt/lists/*
# CUDA 13.4.92 (cuda_nvcc, cudart, crt, libnvvm, cuobjdump, nvdisasm, profiler_api; cccl 13.3.4.3.1), merged into one tree like the mini's ~/cuda-13.4.
# Each tarball is checked against NVIDIA's redist manifest (redistrib_13.4.2.json) sha256.
COPY shim/cuda-redist-13.4.sha256 /tmp/cuda-redist-13.4.sha256
RUN set -eux; mkdir -p ${BUILD_ROOT}/cuda-13.4 /tmp/dl; cd /tmp/dl; \
    while read -r sha p; do curl -fsSL --retry 3 -o "$(basename $p)" "https://developer.download.nvidia.com/compute/cuda/redist/$p"; echo "$sha  $(basename $p)" | sha256sum -c -; done < /tmp/cuda-redist-13.4.sha256; \
    for f in *.tar.xz; do tar -xJf $f; done; \
    for d in */; do cp -a $d/. ${BUILD_ROOT}/cuda-13.4/; done; \
    ln -s lib ${BUILD_ROOT}/cuda-13.4/lib64; rm -rf /tmp/dl; ${BUILD_ROOT}/cuda-13.4/bin/nvcc --version | tail -2
# empty placeholder libraries: Strata's CMake wants CUDA::cublas to exist, nothing is ever linked against them in the Linux tree (the mini's ~/cuda-13.4 has the same two 0-byte files)
RUN touch ${BUILD_ROOT}/cuda-13.4/lib/libcublas.so ${BUILD_ROOT}/cuda-13.4/lib/libcublasLt.so
# the shim's minimal cuBLAS header (the exl3 kernels compile against it; see exl3/build.sh)
COPY shim/include/cublas_v2.h ${BUILD_ROOT}/cuda-13.4/include/cublas_v2.h
ENV PATH=${BUILD_ROOT}/cuda-13.4/bin:${PATH}
WORKDIR ${BUILD_ROOT}

# ---------------------------------------------------------------- fatbins: EXL3 modules + the shim blobs
FROM toolchain AS fatbins
COPY exl3/*.cu ${BUILD_ROOT}/exl3/
COPY exl3/torchstub ${BUILD_ROOT}/exl3/torchstub
COPY exl3/vendor ${BUILD_ROOT}/exl3/vendor
COPY shim/tgcu_builtin.cu shim/tgcu_libc.ptx ${BUILD_ROOT}/shim/
COPY results/verification/fatbin-sha256.txt ${BUILD_ROOT}/verify/fatbin-sha256.txt
COPY shim/prebuilt/SHA256SUMS ${BUILD_ROOT}/verify/prebuilt-SHA256SUMS
COPY docker/build-device.sh /usr/local/bin/build-device.sh
RUN build-device.sh /out/fatbins

# ---------------------------------------------------------------- engine: Strata + patches, configure, device code
FROM toolchain AS engine-source
COPY patches /patches
COPY docker/build-engine.sh /usr/local/bin/build-engine.sh
RUN build-engine.sh source

FROM engine-source AS engine-configure
COPY shim/macbuild.py ${BUILD_ROOT}/shim/macbuild.py
COPY shim/include ${BUILD_ROOT}/shim/include
RUN build-engine.sh configure

FROM engine-configure AS engine-device
ARG JOBS=4
RUN JOBS=${JOBS} build-engine.sh device
# the products the Mac build needs: patched source, configure tree (llama.cpp sources, strata_cmds.txt), the fatbins under their strata-mac paths
RUN set -eux; mkdir -p /out/engine; \
    cp -a ${BUILD_ROOT}/strata /out/engine/strata; \
    cp -a ${BUILD_ROOT}/strata-linux /out/engine/strata-linux; find /out/engine/strata-linux -name .git -prune -exec rm -rf {} +; \
    mkdir -p /out/engine/strata-mac-fatbins; cd ${BUILD_ROOT}/strata-mac && find . -name '*.fatbin' -exec cp --parents {} /out/engine/strata-mac-fatbins/ \;; \
    cp device.ninja ${BUILD_ROOT}/device.log ${BUILD_ROOT}/configure.log /out/engine/; \
    (cd /out/engine/strata-mac-fatbins && find . -type f -name '*.fatbin' | sort | xargs sha256sum) > /out/engine/strata-mac-fatbins.sha256; \
    wc -l /out/engine/strata-mac-fatbins.sha256

# ---------------------------------------------------------------- the bundle
FROM scratch AS bundle-parts
COPY --from=fatbins /out/fatbins /fatbins
COPY --from=engine-device /out/engine /engine

FROM --platform=linux/arm64 ${UBUNTU} AS bundle-assemble
ARG BUILD_ROOT
COPY --from=bundle-parts / /bundle
COPY docker/make-build-info.sh /usr/local/bin/make-build-info.sh
RUN BUILD_ROOT=${BUILD_ROOT} make-build-info.sh /bundle

# `docker buildx build --target bundle -o type=local,dest=out .`  (make)
FROM scratch AS bundle
COPY --from=bundle-assemble /bundle /

# `docker compose run --rm build`: the same files, copied into the bind-mounted ./out
FROM --platform=linux/arm64 ${UBUNTU} AS builder
COPY --from=bundle-assemble /bundle /bundle
CMD ["sh", "-c", "rm -rf /out/fatbins /out/engine && cp -a /bundle/. /out/ && echo \"bundle written to ./out ($(du -sh /out | cut -f1))\""]

# ---------------------------------------------------------------- fetcher (behind docker compose --profile fetch)
FROM --platform=linux/arm64 ${UBUNTU} AS fetcher
ENV DEBIAN_FRONTEND=noninteractive LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3=3.12.3-0ubuntu2.1 python3-venv=3.12.3-0ubuntu2.1 curl=8.5.0-2ubuntu10.15 ca-certificates=20260601~24.04.1 \
 && rm -rf /var/lib/apt/lists/*
# numpy only: pack_td.py / pack_exl3.py need nothing else when run with --no-neon (libexl3cpu is a macOS dylib)
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir numpy==2.5.3
COPY exl3/pack_td.py exl3/pack_exl3.py exl3/td_hdr.py /opt/exl3/
COPY results/pack/exl3_pack.json /opt/exl3_pack.reference.json
COPY docker/fetch.sh docker/models.sha256 /opt/
ENTRYPOINT ["/opt/fetch.sh"]
