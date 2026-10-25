#include "kernels/gemm_internal.h"

#include <cstddef>

namespace game::detail {

namespace {

void nn(int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc, bool accumulate) {
    for (int i = 0; i < m; ++i) {
        float* row = c + static_cast<std::ptrdiff_t>(i) * ldc;
        if (!accumulate) {
            for (int j = 0; j < n; ++j) {
                row[j] = 0.0f;
            }
        }
        for (int kk = 0; kk < k; ++kk) {
            const float scale = a[static_cast<std::ptrdiff_t>(i) * lda + kk];
            const float* source = b + static_cast<std::ptrdiff_t>(kk) * ldb;
            for (int j = 0; j < n; ++j) {
                row[j] += scale * source[j];
            }
        }
    }
}

void nt(int m, int p, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc, bool accumulate) {
    for (int i = 0; i < m; ++i) {
        const float* left = a + static_cast<std::ptrdiff_t>(i) * lda;
        for (int j = 0; j < p; ++j) {
            const float* right = b + static_cast<std::ptrdiff_t>(j) * ldb;
            float sum = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                sum += left[kk] * right[kk];
            }
            float& target = c[static_cast<std::ptrdiff_t>(i) * ldc + j];
            target = accumulate ? target + sum : sum;
        }
    }
}

}

const GemmKernels kGenericGemm{nn, nt, "generic"};

}
