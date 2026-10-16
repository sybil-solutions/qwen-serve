#pragma once
#if defined(__x86_64__) || defined(_M_X64)
#include_next <nmmintrin.h>
#else
#include "immintrin.h"
#endif
