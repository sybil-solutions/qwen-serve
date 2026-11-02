// arm64: no cpuid; every x86 feature query reads as absent
#pragma once
#if defined(__x86_64__) || defined(_M_X64)
#include_next <cpuid.h>
#else
#define __cpuid_count(leaf, sub, a, b, c, d) ((a) = 0, (b) = 0, (c) = 0, (d) = 0)
#define __cpuid(leaf, a, b, c, d) ((a) = 0, (b) = 0, (c) = 0, (d) = 0)
static inline int __get_cpuid_max(unsigned, unsigned*) { return 0; }
#endif
