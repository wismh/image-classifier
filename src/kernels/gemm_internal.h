#pragma once

#include "kernels/gemm.h"

namespace game::detail {

// The kernels of each flavour, defined in their own translation unit (the AVX2 one is compiled with AVX2 enabled, so
// nothing from it may run on a CPU without it: gemm.cpp checks first).
extern const GemmKernels kGenericGemm;
extern const GemmKernels kAvx2Gemm;

}
