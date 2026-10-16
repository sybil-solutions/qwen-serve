// x86 intrinsics on arm64 through SIMDe, so Strata's runtime-dispatched AVX2/AVX-512 kernels compile on Apple silicon.
// The dispatch (cpu_avx2_ok/cpu_avx512_ok) reports false here, so the ggml NEON path is the one that runs.
#pragma once
#if defined(__x86_64__) || defined(_M_X64)
#include_next <immintrin.h>
#else
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx512.h>
#include <simde/x86/fma.h>
#include <simde/x86/f16c.h>
#include <simde/x86/clmul.h>
#ifndef _mm_pause
#define _mm_pause() __asm__ __volatile__("yield")
#endif
#include <stdint.h>
// AVX-512 pieces SIMDe does not provide (scalar, correctness only: these paths are not dispatched on arm64)
#ifndef _cvtu64_mask64
static inline simde__mmask64 _cvtu64_mask64(uint64_t a) { return (simde__mmask64) a; }
#endif
static inline simde__m512i tgcu_mm512_i32gather_epi32(simde__m512i vindex, void const* base, int scale) {
  int32_t ix[16], r[16];
  simde_mm512_storeu_si512(ix, vindex);
  for (int k = 0; k < 16; k++) r[k] = *(const int32_t*) ((const char*) base + (int64_t) ix[k] * scale);
  return simde_mm512_loadu_si512(r);
}
#undef _mm512_i32gather_epi32
#define _mm512_i32gather_epi32(vindex, base, scale) tgcu_mm512_i32gather_epi32(vindex, base, scale)
static inline simde__m512i tgcu_mm512_cvtepu16_epi64(simde__m128i a) {
  uint16_t v[8]; uint64_t r[8];
  simde_mm_storeu_si128((simde__m128i*) v, a);
  for (int k = 0; k < 8; k++) r[k] = v[k];
  return simde_mm512_loadu_si512(r);
}
#undef _mm512_cvtepu16_epi64
#define _mm512_cvtepu16_epi64(a) tgcu_mm512_cvtepu16_epi64(a)
#endif
