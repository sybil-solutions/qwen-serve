// exl3_metal.metal - EXL3 K=3 MUL1 expert MoE compute for the Apple GPU (2560x640 experts: gate/up [2560,640], down [640,2560]).
// Numerics mirror exl3/cpu/exl3_cpu.c + Strata pool.cpp exl3_task: x fp32 -> fp16 -> * suh (fp16 round) -> FWHT128 fp32 (stages 1..64)
// -> * R_SCALE -> fp16 ; GEMV/GEMM with fp16 operands and fp32 accumulation ; had_out = FWHT128 fp32, * INV * svh ; SwiGLU in fp32 ->
// fp16 ; down likewise ; y = sum_pairs weight * expert(x[row]).  Compiled offline without fast-math / fp-contract.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#pragma clang fp contract(off)

#define H 2560u
#define FF 640u
// blob field offsets (exl3_pack.json)
#define O_GT 0ul
#define O_UT 614400ul
#define O_DT 1228800ul
#define O_GSUH 1843200ul
#define O_GSVH 1848320ul
#define O_USUH 1849600ul
#define O_USVH 1854720ul
#define O_DSUH 1856000ul
#define O_DSVH 1857280ul

constant float R_SCALE = 0.088388347648f;
constant float INV = 0.08838834764831845f;

// ---------------------------------------------------------------------------------------------------------------------
inline half cb_mul1(uint s) {
  uint x = s * 0x83DCD12Du;
  uint p = (x & 0x00FF00FFu) + ((x >> 8) & 0x00FF00FFu);      // two 16-bit lanes of 2-byte sums (<= 510)
  uint b = (p * 0x00010001u) >> 16;                          // sum of the 4 bytes (<= 1020)
  half h = as_type<half>((ushort)(b + 0x6400u));
  return fma(h, as_type<half>((ushort)0x1EEE), as_type<half>((ushort)0xC931));
}

// The 8 consecutive states of lane t of a tile (24 uint32 words, stream MSB first, circular; element e's 16-bit window ends at
// bit (e+1)*3), decoded to halves in jj order. Lane t's 8 windows start 3 bits into the 5-byte span that begins at stream byte
// 3t-2 (mod 96), the same layout for every lane, so every shift below is a compile-time constant (32-bit ops only).
inline half cb_state(uint s) { return cb_mul1(s); }
inline void decode8(const device uint* w, uint t, thread half* v) {
  const uint sb = (3u * t + 94u) % 96u;
  const uint wi = sb >> 2, sh = (sb & 3u) * 8u;
  const uint i1 = wi == 23u ? 0u : wi + 1u, i2 = i1 == 23u ? 0u : i1 + 1u;
  const uint w0 = w[wi], w1 = w[i1], w2 = w[i2];
  const uint hi = sh ? ((w0 << sh) | (w1 >> (32u - sh))) : w0;     // span bytes 0..3
  const uint nx = sh ? ((w1 << sh) | (w2 >> (32u - sh))) : w1;     // span bytes 4..7
  v[0] = cb_state((hi >> 13) & 0xFFFFu);
  v[1] = cb_state((hi >> 10) & 0xFFFFu);
  v[2] = cb_state((hi >> 7) & 0xFFFFu);
  v[3] = cb_state((hi >> 4) & 0xFFFFu);
  v[4] = cb_state((hi >> 1) & 0xFFFFu);
  v[5] = cb_state(((hi << 2) | (nx >> 30)) & 0xFFFFu);
  v[6] = cb_state(((hi << 5) | (nx >> 27)) & 0xFFFFu);
  v[7] = cb_state(((hi << 8) | (nx >> 24)) & 0xFFFFu);
}

// FWHT128 over a 128-thread threadgroup (thread i holds element i): stages 1..16 by simd shuffles, 32 and 64 through TG memory.
// Same operations in the same order as the sequential butterfly -> bitwise equal.
inline float fwht128(float v, uint i, threadgroup float* buf) {
  for (uint st = 1u; st < 32u; st <<= 1) {
    float o = simd_shuffle_xor(v, (ushort)st);
    v = (i & st) ? (o - v) : (v + o);
  }
  for (uint st = 32u; st <= 64u; st <<= 1) {
    buf[i] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = buf[i ^ st];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v = (i & st) ? (o - v) : (v + o);
  }
  return v;
}

// permuted-input layout of the GEMV: element e of a row -> ((e/16)*4 + q)*4 + slot with q = ((e%16)%8)/2, slot = (e%2) | (e%16/8)*2
inline uint xq_index(uint e) {
  uint i = e >> 4, r = e & 15u;
  return ((i * 4u + ((r & 7u) >> 1)) << 2) + (r & 1u) + ((r >> 3) << 1);
}

// ---------------------------------------------------------------------------------------------------------------------
// DECODE shape: unit = one distinct expert with up to mrx rows (slots).
struct UnitD { device const uchar* blob; int rows[8]; };       // rows[slot]: row of x, -1 = unused
struct PairD { uint u; uint slot; float w; uint pad; };         // pair: unit, slot, weight
struct DP { uint mrx; uint nu; uint pad0; uint pad1; };

// A: rotate inputs for gate (proj 0) and up (proj 1). grid (20 blocks, nu*mrx, 2), TG 128. xq [proj][u][slot][H] permuted.
kernel void rot_in_dec(device const float* x [[buffer(0)]], device const UnitD* units [[buffer(1)]], device half* xq [[buffer(2)]],
                       constant DP& p [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  threadgroup float buf[128];
  const uint blk = tg.x, u = tg.y / p.mrx, slot = tg.y % p.mrx, proj = tg.z;
  const int row = units[u].rows[slot];
  float v = 0.f;
  if (row >= 0) {
    half a = (half)x[(ulong)row * H + blk * 128u + i];
    half s = ((device const half*)(units[u].blob + (proj ? O_USUH : O_GSUH)))[blk * 128u + i];
    v = (float)(half)((float)a * (float)s);
  }
  v = fwht128(v, i, buf);
  xq[(((ulong)proj * p.nu + u) * p.mrx + slot) * H + xq_index(blk * 128u + i)] = (half)(v * R_SCALE);
}

// B: fused trellis decode + GEMV. One simdgroup (32 lanes = the 32 lane-groups of a tile) per (16-column tile, unit, proj).
// kind 0: gate/up (k=2560, n=640) grid (40, nu, 2); kind 1: down (k=640, n=2560) grid (160, nu, 1).
struct GP { uint kind; uint pad[3]; };
template <uint MR>
kernel void gemv_dec(device const UnitD* units [[buffer(0)]], device const half* xin [[buffer(1)]], device float* yout [[buffer(2)]],
                     constant DP& p [[buffer(3)]], constant GP& g [[buffer(4)]],
                     uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  const uint k = g.kind ? FF : H, n = g.kind ? H : FF, kt = k / 16u, nt = n / 16u;
  const uint j = tg.x, u = tg.y, proj = tg.z;
  device const uchar* blob = units[u].blob;
  const device uint* tr = (const device uint*)(blob + (g.kind ? O_DT : (proj ? O_UT : O_GT)));
  const ulong ub = g.kind ? (ulong)u : ((ulong)proj * p.nu + u);
  device const half* x = xin + ub * MR * k;
  device float* y = yout + ub * MR * n;
  const uint c = lane >> 2, q = lane & 3u;
  float accA[MR], accB[MR];
  for (uint r = 0; r < MR; r++) { accA[r] = 0.f; accB[r] = 0.f; }
  for (uint i = 0; i < kt; i++) {
    half w[8];
    decode8(tr + ((ulong)i * nt + j) * 24u, lane, w);
    for (uint r = 0; r < MR; r++) {
      half4 xv = *(device const half4*)(x + (ulong)r * k + (i * 4u + q) * 4u);
      float a = accA[r], b = accB[r];
      a += (float)w[0] * (float)xv[0]; a += (float)w[1] * (float)xv[1]; a += (float)w[2] * (float)xv[2]; a += (float)w[3] * (float)xv[3];
      b += (float)w[4] * (float)xv[0]; b += (float)w[5] * (float)xv[1]; b += (float)w[6] * (float)xv[2]; b += (float)w[7] * (float)xv[3];
      accA[r] = a; accB[r] = b;
    }
  }
  for (uint r = 0; r < MR; r++) {
    float a = accA[r], b = accB[r];
    a += simd_shuffle_xor(a, (ushort)1); a += simd_shuffle_xor(a, (ushort)2);
    b += simd_shuffle_xor(b, (ushort)1); b += simd_shuffle_xor(b, (ushort)2);
    if (q == 0) { y[(ulong)r * n + 16u * j + c] = a; y[(ulong)r * n + 16u * j + c + 8u] = b; }
  }
}
template [[host_name("gemv_dec_1")]] kernel void gemv_dec<1>(device const UnitD*, device const half*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_dec_2")]] kernel void gemv_dec<2>(device const UnitD*, device const half*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_dec_4")]] kernel void gemv_dec<4>(device const UnitD*, device const half*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_dec_8")]] kernel void gemv_dec<8>(device const UnitD*, device const half*, device float*, constant DP&, constant GP&, uint3, uint);

// C: had_out(gate), had_out(up), SwiGLU -> fp16, rotate for down. grid (5, nu*mrx), TG 128. zq [u][slot][FF] permuted.
kernel void mid_dec(device const UnitD* units [[buffer(0)]], device const float* yg [[buffer(1)]], device half* zq [[buffer(2)]],
                    constant DP& p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  threadgroup float buf[128];
  const uint blk = tg.x, u = tg.y / p.mrx, slot = tg.y % p.mrx;
  device const uchar* blob = units[u].blob;
  const ulong rowi = (ulong)u * p.mrx + slot, plane = (ulong)p.nu * p.mrx * FF;
  const uint col = blk * 128u + i;
  float vg = fwht128(yg[rowi * FF + col], i, buf);
  float vu = fwht128(yg[plane + rowi * FF + col], i, buf);
  float gg = vg * INV * (float)((device const half*)(blob + O_GSVH))[col];
  float uu = vu * INV * (float)((device const half*)(blob + O_USVH))[col];
  half z = (half)(gg / (1.f + precise::exp(-gg)) * uu);
  float v = (float)(half)((float)z * (float)((device const half*)(blob + O_DSUH))[col]);
  v = fwht128(v, i, buf);
  zq[rowi * FF + xq_index(col)] = (half)(v * R_SCALE);
}

// E: had_out(down) and the weighted sum over pairs, per output row in pair order. grid (20, nrows), TG 128. out [row][H].
kernel void comb_dec(device const UnitD* units [[buffer(0)]], device const float* yd [[buffer(1)]], device const PairD* pairs [[buffer(2)]],
                     device const uint* rowptr [[buffer(3)]], device float* out [[buffer(4)]], constant DP& p [[buffer(5)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  threadgroup float buf[128];
  const uint blk = tg.x, row = tg.y, col = blk * 128u + i;
  float acc = 0.f;
  for (uint pi = rowptr[row]; pi < rowptr[row + 1]; pi++) {
    PairD pr = pairs[pi];
    float v = fwht128(yd[((ulong)pr.u * p.mrx + pr.slot) * H + col], i, buf);
    float o = v * INV * (float)((device const half*)(units[pr.u].blob + O_DSVH))[col];
    acc += pr.w * o;
  }
  out[(ulong)row * H + col] = acc;
}

// ---------------------------------------------------------------------------------------------------------------------
// PREFILL shape: experts with many rows. Rows of the chunk are laid out expert after expert (row0 per unit).
struct UnitP { device const uchar* blob; uint row0; uint nrows; };
struct PP { uint nrows; uint cap; uint pad0; uint pad1; };      // rows in chunk; row capacity (plane stride of the rotated inputs)

// rotate inputs per chunk row. grid (20, nrows, 2), TG 128. A [proj][cap][H] plain fp16.
kernel void rot_in_pf(device const float* x [[buffer(0)]], device const UnitP* units [[buffer(1)]], device const uint* unit_of_row [[buffer(2)]],
                      device const int* src_row [[buffer(3)]], device half* A [[buffer(4)]], constant PP& p [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  threadgroup float buf[128];
  const uint blk = tg.x, pr = tg.y, proj = tg.z, col = blk * 128u + i;
  half a = (half)x[(ulong)src_row[pr] * H + col];
  half s = ((device const half*)(units[unit_of_row[pr]].blob + (proj ? O_USUH : O_GSUH)))[col];
  float v = fwht128((float)(half)((float)a * (float)s), i, buf);
  A[((ulong)proj * p.cap + pr) * H + col] = (half)(v * R_SCALE);
}

// decode a projection of every chunk unit into the tiled fp16 layout W[u][tile(i*nt+j)][row*16+col] (512 B per tile).
// grid x = ntiles*32 (tile = x>>5, lane = x&31), y = unit; which: 0 gate, 1 up, 2 down.
kernel void decode_w(device const UnitP* units [[buffer(0)]], device half* W [[buffer(1)]], constant uint& which [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
  const uint ntiles = (H / 16u) * (FF / 16u), tile = gid.x >> 5, t = gid.x & 31u;
  if (tile >= ntiles) return;
  const uint u = gid.y;
  const device uint* w = (const device uint*)(units[u].blob + (which == 2 ? O_DT : (which ? O_UT : O_GT))) + (ulong)tile * 24u;
  half v[8];
  decode8(w, t, v);
  const uint c = t >> 2, q = t & 3u;
  device half* o = W + ((ulong)u * ntiles + tile) * 256u;
  for (uint jj = 0; jj < 8u; jj++) {
    uint row = 2u * q + (jj & 1u) + ((jj & 2u) ? 8u : 0u), col = c + ((jj & 4u) ? 8u : 0u);
    o[row * 16u + col] = v[jj];
  }
}

// C[nv rows, N] fp32 = A[rows, K] fp16 x W[K, N] (tiled fp16, per unit) ; tile 32 x 64, K step 32, 4 simdgroups.
struct GT { uint u; uint r0; uint nv; uint pad; };
template <uint K, uint N, bool FROMB>
inline void gemm_body(device const half* A, device const half* Wu, device float* C, uint r0, uint nv, uint n0, uint t, uint sg,
                      threadgroup half* As, threadgroup half* Bs, threadgroup float* Cs) {
  constexpr uint NT = N / 16u;
  const uint sgm = sg >> 1, sgn = sg & 1u;
  simdgroup_float8x8 acc[2][4];
  for (uint a = 0; a < 2; a++) for (uint b = 0; b < 4; b++) acc[a][b] = simdgroup_float8x8(0.f);
  const uint arow = t >> 2, apart = t & 3u;                // A tile: 32 rows x 32 k: 4 x half8 per row
  const uint bt = t >> 4, br = t & 15u;                    // B tile: 8 16x16 tiles, one 16-col row (32 B) per thread
  const uint bki = bt >> 2, bnj = bt & 3u;
  for (uint k0 = 0; k0 < K; k0 += 32u) {
    *(threadgroup half4*)(As + arow * 32u + apart * 8u) = *(device const half4*)(A + (ulong)(r0 + arow) * K + k0 + apart * 8u);
    *(threadgroup half4*)(As + arow * 32u + apart * 8u + 4u) = *(device const half4*)(A + (ulong)(r0 + arow) * K + k0 + apart * 8u + 4u);
    device const half* bsrc = Wu + ((ulong)((k0 >> 4) + bki) * NT + ((n0 >> 4) + bnj)) * 256u + br * 16u;
    *(threadgroup half4*)(Bs + (bki * 16u + br) * 64u + bnj * 16u) = *(device const half4*)bsrc;
    *(threadgroup half4*)(Bs + (bki * 16u + br) * 64u + bnj * 16u + 4u) = *(device const half4*)(bsrc + 4);
    *(threadgroup half4*)(Bs + (bki * 16u + br) * 64u + bnj * 16u + 8u) = *(device const half4*)(bsrc + 8);
    *(threadgroup half4*)(Bs + (bki * 16u + br) * 64u + bnj * 16u + 12u) = *(device const half4*)(bsrc + 12);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 32u; kk += 8u) {
      simdgroup_half8x8 a0, a1, b0, b1, b2, b3;
      simdgroup_load(a0, As + (sgm * 16u) * 32u + kk, 32);
      simdgroup_load(a1, As + (sgm * 16u + 8u) * 32u + kk, 32);
      simdgroup_load(b0, Bs + kk * 64u + sgn * 32u, 64);
      simdgroup_load(b1, Bs + kk * 64u + sgn * 32u + 8u, 64);
      simdgroup_load(b2, Bs + kk * 64u + sgn * 32u + 16u, 64);
      simdgroup_load(b3, Bs + kk * 64u + sgn * 32u + 24u, 64);
      simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]); simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
      simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]); simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);
      simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]); simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
      simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]); simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint a = 0; a < 2; a++) for (uint b = 0; b < 4; b++)
    simdgroup_store(acc[a][b], Cs + (sgm * 16u + a * 8u) * 64u + sgn * 32u + b * 8u, 64);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = t; e < 32u * 16u; e += 128u) {              // 32 rows x 16 float4
    uint r = e >> 4, c4 = e & 15u;
    if (r < nv) *(device float4*)(C + (ulong)(r0 + r) * N + n0 + c4 * 4u) = *(threadgroup float4*)(Cs + r * 64u + c4 * 4u);
  }
}

// gate+up: grid (10 n tiles, ntiles, 2 proj); A [proj][cap][H], W [proj? Wg : Wu][u], C [proj][cap][FF]
kernel void gemm_gu(device const half* A [[buffer(0)]], device const half* Wg [[buffer(1)]], device const half* Wu [[buffer(2)]],
                    device float* C [[buffer(3)]], device const GT* tiles [[buffer(4)]], constant PP& p [[buffer(5)]],
                    uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup half As[32 * 32]; threadgroup half Bs[32 * 64]; threadgroup float Cs[32 * 64];
  const GT T = tiles[tg.y];
  const uint proj = tg.z;
  device const half* W = (proj ? Wu : Wg) + (ulong)T.u * (H * FF);
  gemm_body<H, FF, true>(A + (ulong)proj * p.cap * H, W, C + (ulong)proj * p.cap * FF, T.r0, T.nv, tg.x * 64u, t, sg, As, Bs, Cs);
}
// down: grid (40 n tiles, ntiles, 1); A = Z [cap][FF], W = Wd[u], C = Yd [cap][H]
kernel void gemm_d(device const half* A [[buffer(0)]], device const half* Wd [[buffer(1)]], device float* C [[buffer(2)]],
                   device const GT* tiles [[buffer(3)]],
                   uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup half As[32 * 32]; threadgroup half Bs[32 * 64]; threadgroup float Cs[32 * 64];
  const GT T = tiles[tg.y];
  gemm_body<FF, H, true>(A, Wd + (ulong)T.u * (H * FF), C, T.r0, T.nv, tg.x * 64u, t, sg, As, Bs, Cs);
}

// had_out(gate/up) + SwiGLU + rotate for down. grid (5, nrows). Cg [proj][cap][FF] fp32 -> Z [cap][FF] fp16 (plain).
kernel void mid_pf(device const UnitP* units [[buffer(0)]], device const uint* unit_of_row [[buffer(1)]], device const float* Cgu [[buffer(2)]],
                   device half* Z [[buffer(3)]], constant PP& p [[buffer(4)]],
                   uint2 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  threadgroup float buf[128];
  const uint blk = tg.x, pr = tg.y, col = blk * 128u + i;
  device const uchar* blob = units[unit_of_row[pr]].blob;
  float vg = fwht128(Cgu[(ulong)pr * FF + col], i, buf);
  float vu = fwht128(Cgu[((ulong)p.cap + pr) * FF + col], i, buf);
  float gg = vg * INV * (float)((device const half*)(blob + O_GSVH))[col];
  float uu = vu * INV * (float)((device const half*)(blob + O_USVH))[col];
  half z = (half)(gg / (1.f + precise::exp(-gg)) * uu);
  float v = fwht128((float)(half)((float)z * (float)((device const half*)(blob + O_DSUH))[col]), i, buf);
  Z[(ulong)pr * FF + col] = (half)(v * R_SCALE);
}

// had_out(down) + weighted accumulation into y (+=) per touched output row, pairs of the chunk in order. grid (20, ntouched).
struct CP { uint pr; float w; };
kernel void comb_pf(device const UnitP* units [[buffer(0)]], device const uint* unit_of_row [[buffer(1)]], device const float* Yd [[buffer(2)]],
                    device const uint* touched [[buffer(3)]], device const uint* tptr [[buffer(4)]], device const CP* cps [[buffer(5)]],
                    device float* y [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  threadgroup float buf[128];
  const uint blk = tg.x, ti = tg.y, col = blk * 128u + i;
  float acc = 0.f;
  for (uint c = tptr[ti]; c < tptr[ti + 1]; c++) {
    CP cp = cps[c];
    float v = fwht128(Yd[(ulong)cp.pr * H + col], i, buf);
    float o = v * INV * (float)((device const half*)(units[unit_of_row[cp.pr]].blob + O_DSVH))[col];
    acc += cp.w * o;
  }
  const ulong yi = (ulong)touched[ti] * H + col;
  y[yi] = y[yi] + acc;
}

// fp16-I/O prefill (pf500): the same rotation with the input rows already fp16 (the kernel rounds x to fp16 first anyway: bit-identical to rot_in_pf on fp16-representable x)
kernel void rot_in_pf_h(device const half* x [[buffer(0)]], device const UnitP* units [[buffer(1)]], device const uint* unit_of_row [[buffer(2)]],
                        device const int* src_row [[buffer(3)]], device half* A [[buffer(4)]], constant PP& p [[buffer(5)]],
                        uint3 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  threadgroup float buf[128];
  const uint blk = tg.x, pr = tg.y, proj = tg.z, col = blk * 128u + i;
  half a = x[(ulong)src_row[pr] * H + col];
  half s = ((device const half*)(units[unit_of_row[pr]].blob + (proj ? O_USUH : O_GSUH)))[col];
  float v = fwht128((float)(half)((float)a * (float)s), i, buf);
  A[((ulong)proj * p.cap + pr) * H + col] = (half)(v * R_SCALE);
}

// the fp32 accumulated rows -> fp16 (what the host's clamp + _Float16 cast did): clamp to +-65504 (NaN passes), round to nearest even
kernel void y_f2h(device const float* y [[buffer(0)]], device half* yh [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) {
    float v = y[i];
    v = v > 65504.0f ? 65504.0f : (v < -65504.0f ? -65504.0f : v);
    yh[i] = (half)v;
  }
}
