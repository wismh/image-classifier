#include "kernels/gemm.h"

#include "kernels/gemm_internal.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#include <intrin.h>
#endif

namespace game {

namespace {

#if defined(PETS_HAVE_AVX2_KERNELS)
bool cpu_has_avx2_and_fma() {
#if defined(_MSC_VER)
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) {
        return false;
    }
    __cpuid(info, 1);
    const bool fma = (info[2] & (1 << 12)) != 0;
    const bool osxsave = (info[2] & (1 << 27)) != 0;
    const bool avx = (info[2] & (1 << 28)) != 0;
    if (!fma || !osxsave || !avx) {
        return false;
    }
    // The operating system must save the wide registers across thread switches.
    if ((_xgetbv(0) & 0x6) != 0x6) {
        return false;
    }
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
#else
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
}
#endif

}

const GemmKernels& generic_gemm() {
    return detail::kGenericGemm;
}

const GemmKernels* avx2_gemm() {
#if defined(PETS_HAVE_AVX2_KERNELS)
    static const bool supported = cpu_has_avx2_and_fma();
    return supported ? &detail::kAvx2Gemm : nullptr;
#else
    return nullptr;
#endif
}

const GemmKernels& best_gemm() {
    static const GemmKernels& best = avx2_gemm() != nullptr ? *avx2_gemm() : generic_gemm();
    return best;
}

}
