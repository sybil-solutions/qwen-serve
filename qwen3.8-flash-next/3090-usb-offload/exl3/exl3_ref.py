"""Numpy bit-exact reference for the EXL3 (exllamav3 v1.x) trellis decode, integer K 1..8, codebooks 3INST/MCG/MUL1.

Format (W is (k=in, n=out), y = x @ W):
  trellis uint16 [k/16, n/16, 16K]   one 16x16 tile per (i, j); 256*K bits per tile
  suh fp16 [k], svh fp16 [n]         sign/scale vectors
Tile stream: uint32 words = little-endian pairs of the uint16 storage, bit 0 = MSB of word 0. Element t (0..255)
decodes the 16-bit window ending at bit (t+1)*K (circular in the tile) and lands at tensor-core position perm[t].
W_orig = diag(suh) H128 W_inner H128 diag(svh) (normalized Sylvester Hadamard per 128 block).

Every fp16 operation below is one IEEE op rounded to fp16: numpy evaluates fp16 add/sub/mul in fp32 and rounds,
which equals the single fp16 rounding (24 >= 2*11+2 bits). Butterfly stage order matches the CUDA kernels, so the
reconstruct, fused-Hadamard reconstruct and input Hadamard outputs are compared bit for bit.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np

CB_3INST, CB_MCG, CB_MUL1 = 0, 1, 2
MCG_MULT, MUL1_MULT = 0xCBAC1FED, 0x83DCD12D
R_SCALE_F32 = np.float32(0.088388347648)                # had_hf_r_128_inner's r_scale argument
R_SCALE_F16 = np.float16(np.float32(0.08838834764831845))  # reconstruct_had_tile: __float2half2_rn(r_scale)


def _f16(bits) -> np.ndarray: return np.asarray(bits, dtype=np.uint16).view(np.float16)


@lru_cache
def tensor_core_perm() -> np.ndarray:
    perm = np.zeros(256, np.int64)
    for t in range(32):
        r0 = (t % 4) * 2
        c0 = t // 4
        for j, (r, c) in enumerate([(r0, c0), (r0 + 1, c0), (r0 + 8, c0), (r0 + 9, c0),
                                    (r0, c0 + 8), (r0 + 1, c0 + 8), (r0 + 8, c0 + 8), (r0 + 9, c0 + 8)]):
            perm[t * 8 + j] = r * 16 + c
    return perm


@lru_cache
def codebook_lut(cb: int) -> np.ndarray:
    """fp16 value of every 16-bit state, as codebook.cuh decodes it."""
    s = np.arange(65536, dtype=np.uint64)
    if cb == CB_MUL1:
        x = (s * MUL1_MULT) & 0xFFFFFFFF
        bsum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
        h = (1024 + bsum).astype(np.float64)                              # fp16 0x6400 + bsum, exact
        # __hfma2: exact h*inv + bias (fits float64), one rounding to fp16
        return (h * float(_f16(0x1EEE)) + float(_f16(0xC931))).astype(np.float16)
    x = ((s * 89226354 + 64248484) if cb == CB_3INST else (s * MCG_MULT)) & 0xFFFFFFFF
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60                                     # lop3 0x6a
    return (_f16(x & 0xFFFF).astype(np.float64) + _f16(x >> 16).astype(np.float64)).astype(np.float16)


@lru_cache
def _windows(K: int) -> np.ndarray:
    nbits = 256 * K
    b0 = (np.arange(256) * K + K - 16 + nbits) % nbits
    return (b0[:, None] + np.arange(16)[None, :]) % nbits                  # [256, 16] bit positions, MSB first


def decode_states(trellis: np.ndarray, K: int) -> np.ndarray:
    """trellis uint16 [..., 16K] -> states int64 [..., 256] in stream order."""
    t = trellis.astype(np.uint32)
    words = t[..., 0::2] | (t[..., 1::2] << 16)
    bits = ((words[..., None] >> np.arange(31, -1, -1, dtype=np.uint32)) & 1).reshape(*trellis.shape[:-1], 256 * K)
    win = bits[..., _windows(K)].astype(np.int64)                         # [..., 256, 16]
    return (win << np.arange(15, -1, -1)).sum(-1)


def reconstruct(trellis: np.ndarray, K: int, cb: int) -> np.ndarray:
    """trellis [k/16, n/16, 16K] -> W_inner fp16 [k, n]."""
    tk, tn, _ = trellis.shape
    flat = trellis.reshape(tk * tn, 16 * K)
    lut = codebook_lut(cb)
    vals = np.concatenate([lut[decode_states(flat[s:s + 2048], K)] for s in range(0, tk * tn, 2048)])
    tile = np.empty_like(vals)
    tile[:, tensor_core_perm()] = vals
    return tile.reshape(tk, tn, 16, 16).transpose(0, 2, 1, 3).reshape(tk * 16, tn * 16)


def _bfly(x: np.ndarray, axis: int, stride: int) -> np.ndarray:
    """One Sylvester stage along axis (length 128): (a, b) -> (a + b, a - b) at index pairs i, i + stride."""
    x = np.moveaxis(x, axis, -1)
    s = x.shape
    v = x.reshape(*s[:-1], 128 // (2 * stride), 2, stride)
    a, b = v[..., 0, :], v[..., 1, :]
    out = np.stack([a + b, a - b], axis=-2).reshape(s)
    return np.moveaxis(out, -1, axis)


def had_in(A: np.ndarray, suh: np.ndarray) -> np.ndarray:
    """had_hf_r_128_inner<pre_scale>: fp16 A [m, k] * suh -> fp32 FWHT per 128 -> * r_scale -> fp16."""
    m, k = A.shape
    v = (A * suh[None, :]).astype(np.float32).reshape(m, k // 128, 128)
    for st in (1, 2, 4, 8, 16, 32, 64): v = _bfly(v, 2, st)
    return (v * R_SCALE_F32).astype(np.float16).reshape(m, k)


def reconstruct_had(W_inner: np.ndarray, suh: np.ndarray, svh: np.ndarray) -> np.ndarray:
    """reconstruct_had_tile on a decoded W_inner fp16 [k, n] (k, n multiples of 128)."""
    k, n = W_inner.shape
    w = W_inner.reshape(k // 128, 128, n // 128, 128)                      # [kb, R, nb, c]
    # column transform (along R), fp16: stride 1, 2, scale, then lane strides 4..64
    w = _bfly(w, 1, 1)
    w = _bfly(w, 1, 2) * R_SCALE_F16
    for st in (4, 8, 16, 32, 64): w = _bfly(w, 1, st)
    # row transform (along c): stride 1, 2 in fp32, round to fp16, scale, then fp16 lane strides 4..64
    w = _bfly(_bfly(w.astype(np.float32), 3, 1), 3, 2).astype(np.float16) * R_SCALE_F16
    for st in (4, 8, 16, 32, 64): w = _bfly(w, 3, st)
    w = w * suh.reshape(k // 128, 128, 1, 1)
    w = w * svh.reshape(1, 1, n // 128, 128)
    return w.reshape(k, n)


def weight_orig_f64(W_inner: np.ndarray, suh: np.ndarray, svh: np.ndarray) -> np.ndarray:
    """Exact-arithmetic W_orig (float64), for error statistics of the fp16 pipelines."""
    k, n = W_inner.shape
    h = np.ones((1, 1))
    while h.shape[0] < 128: h = np.block([[h, h], [h, -h]])
    h /= np.sqrt(128)
    w = W_inner.astype(np.float64).reshape(k // 128, 128, n // 128, 128)
    w = np.einsum("ir,arbc->aibc", h, w, optimize=True) @ h   # H128 on both sides, 2 GEMMs (4-operand einsum: ~11 s)
    w = w.reshape(k, n)
    return w * suh.astype(np.float64)[:, None] * svh.astype(np.float64)[None, :]


def gemv_f64(A_had: np.ndarray, W_inner: np.ndarray, svh: np.ndarray) -> np.ndarray:
    """exl3_gemv_kernel in exact arithmetic from the same fp16 operands: C = had_out(A_had @ W_inner), had_out = H128
    per 128 output columns, / sqrt(128), * svh. Equals A @ W_orig up to the fp16 rounding of A_had."""
    m, n = A_had.shape[0], W_inner.shape[1]
    y = (A_had.astype(np.float64) @ W_inner.astype(np.float64)).reshape(m, n // 128, 128)
    for st in (1, 2, 4, 8, 16, 32, 64): y = _bfly(y, 2, st)
    return y.reshape(m, n) / np.sqrt(128) * svh.astype(np.float64)[None, :]


# --- encoder side (M3): exllamav3 quantize_tiles_kernel (generic Viterbi, L = 256) and pack_trellis_kernel -----------
def _brev5(v: np.ndarray) -> np.ndarray:
    return sum(((v >> b) & 1) << (4 - b) for b in range(5))


@lru_cache
def _viterbi_tables(K: int, cb: int):
    """Per (k, out_edge): the 16-bit state (k << (16 - K)) | out_edge, its in-edge state >> K, its fp16 value; the
    argmin_cost tie rank of every edge (bit-reversed warp, bit-reversed lane, then 1024-state stripe)."""
    edges = 65536 >> K
    e = np.arange(edges, dtype=np.int64)
    states = (np.arange(1 << K, dtype=np.int64)[:, None] << (16 - K)) | e[None, :]
    v = e & 1023
    rank = (_brev5(v >> 5) << 10) | (_brev5(v & 31) << 5) | (e >> 10)
    return states >> K, codebook_lut(cb)[states].astype(np.float64), rank


def viterbi_tiles(tiles: np.ndarray, K: int, cb: int) -> np.ndarray:
    """quantize_tiles_kernel<K, cb, 256> in numpy: fp32 tiles [T, 256] (tensor-core order) -> uint16 states [T, 256].
    Each trellis step is evaluated as the kernel does: w = fp16(x); dh = hsub(dec, w); cost = hmul(dh, dh) (first
    step) or hfma(dh, dh, prev[in_edge]), one fp16 rounding each (exact in float64 before the rounding, except an
    fma whose product lies > 2^53 below the addend, where both round to the addend); the first candidate k wins ties
    (strict __hlt); argmin over the final costs uses the kernel's packed (cost bits, rank) key; two passes (roll
    L / 2 unconstrained, trace back to position 0, then roll 0 constrained to that state), tail-biting."""
    T, L = tiles.shape
    assert L == 256
    in_edge, dec, rank = _viterbi_tables(K, cb)
    edges = 65536 >> K
    w = tiles.astype(np.float16).astype(np.float64)
    hist = np.zeros((T, L, edges), np.uint16)
    rows = np.arange(T)[:, None]
    cols = np.arange(edges)[None, :]

    def forward(roll, pre):
        cost = None
        for i in range(L):
            ri = (i + roll) % L
            dh = (dec[None] - w[:, ri, None, None]).astype(np.float16).astype(np.float64)   # [T, 2^K, edges]
            if cost is None:
                err = (dh * dh).astype(np.float16)
                if pre is not None:
                    err[in_edge[None] != pre[:, None, None]] = np.inf
            else:
                err = (dh * dh + cost.astype(np.float64)[:, in_edge]).astype(np.float16)
            kbest = err.argmin(axis=1)                                                       # first min = strict <
            cost = err[rows, kbest, cols]
            hist[:, ri] = in_edge[kbest, cols]
        return cost

    def backward(roll, edge, out):
        for t in range(T):
            e = int(edge[t])
            for i in range(L - 1, -1, -1):
                ri = (i + roll) % L
                prev = int(hist[t, ri, e])
                if out is not None: out[t, ri] = ((prev << K) | e) & 0xFFFF
                e = prev
                if out is None and ri == 0: break
            edge[t] = e
        return edge

    with np.errstate(over="ignore", invalid="ignore"):
        cost = forward(L // 2, None)
        bits = cost.view(np.uint16).astype(np.int64)
        key = (bits << 16) | rank[None, :]
        best = key.argmin(axis=1)
        end = np.where(key[np.arange(T), best] >= (0x7C00 << 16), 0, best)
        end = backward(L // 2, end.copy(), None)
        forward(0, end)
        out = np.zeros((T, L), np.uint16)
        backward(0, end.copy(), out)
    return out


def pack_trellis(states: np.ndarray, K: int) -> np.ndarray:
    """uint16 states [..., 256] (stream order) -> trellis uint16 [..., 16K]: element t's low K bits at stream bits
    [tK, (t+1)K), MSB first; uint32 words = little-endian uint16 pairs (the layout decode_states reads)."""
    s = states.astype(np.uint32) & ((1 << K) - 1)
    bits = ((s[..., None] >> np.arange(K - 1, -1, -1, dtype=np.uint32)) & 1).reshape(*s.shape[:-1], 8 * K, 32)
    words = (bits << np.arange(31, -1, -1, dtype=np.uint32)).sum(-1, dtype=np.uint32)
    out = np.empty((*s.shape[:-1], 16 * K), np.uint16)
    out[..., 0::2] = words & 0xFFFF
    out[..., 1::2] = words >> 16
    return out
