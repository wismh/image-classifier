// Compiled with AVX2 and FMA enabled (see CMakeLists.txt); only run after gemm.cpp has checked the CPU.

#include "kernels/gemm_internal.h"

#include <immintrin.h>

#include <algorithm>
#include <cstddef>

namespace game::detail {

namespace {

constexpr int kRows = 6;      // rows of C a block of nn() computes at once
constexpr int kColumns = 16;  // columns of C (two vectors of 8)

// R rows x 16 columns of C, kept in registers over the whole walk along k: per step two loads of B and R broadcasts of
// A
// feed 2R multiply-adds.
template<int R>
void nn_block(int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc, bool accumulate) {
    __m256 low[R];
    __m256 high[R];
    for (int r = 0; r < R; ++r) {
        low[r] = accumulate ? _mm256_loadu_ps(c + static_cast<std::ptrdiff_t>(r) * ldc) : _mm256_setzero_ps();
        high[r] = accumulate ? _mm256_loadu_ps(c + static_cast<std::ptrdiff_t>(r) * ldc + 8) : _mm256_setzero_ps();
    }
    for (int kk = 0; kk < k; ++kk) {
        const __m256 b_low = _mm256_loadu_ps(b + static_cast<std::ptrdiff_t>(kk) * ldb);
        const __m256 b_high = _mm256_loadu_ps(b + static_cast<std::ptrdiff_t>(kk) * ldb + 8);
        for (int r = 0; r < R; ++r) {
            const __m256 scale = _mm256_broadcast_ss(a + static_cast<std::ptrdiff_t>(r) * lda + kk);
            low[r] = _mm256_fmadd_ps(scale, b_low, low[r]);
            high[r] = _mm256_fmadd_ps(scale, b_high, high[r]);
        }
    }
    for (int r = 0; r < R; ++r) {
        _mm256_storeu_ps(c + static_cast<std::ptrdiff_t>(r) * ldc, low[r]);
        _mm256_storeu_ps(c + static_cast<std::ptrdiff_t>(r) * ldc + 8, high[r]);
    }
}

// R rows x 8 columns, for the columns left over after the blocks of 16.
template<int R>
void nn_block8(int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc, bool accumulate) {
    __m256 acc[R];
    for (int r = 0; r < R; ++r) {
        acc[r] = accumulate ? _mm256_loadu_ps(c + static_cast<std::ptrdiff_t>(r) * ldc) : _mm256_setzero_ps();
    }
    for (int kk = 0; kk < k; ++kk) {
        const __m256 values = _mm256_loadu_ps(b + static_cast<std::ptrdiff_t>(kk) * ldb);
        for (int r = 0; r < R; ++r) {
            acc[r] = _mm256_fmadd_ps(_mm256_broadcast_ss(a + static_cast<std::ptrdiff_t>(r) * lda + kk), values, acc[r]);
        }
    }
    for (int r = 0; r < R; ++r) {
        _mm256_storeu_ps(c + static_cast<std::ptrdiff_t>(r) * ldc, acc[r]);
    }
}

using NnBlock = void (*)(int, const float*, int, const float*, int, float*, int, bool);

// The block function for `rows` rows (1..kRows).
template<int Width>
NnBlock nn_block_for(int rows) {
    static constexpr NnBlock kWide[kRows] = {nn_block<1>, nn_block<2>, nn_block<3>, nn_block<4>, nn_block<5>, nn_block<6>};
    static constexpr NnBlock kNarrow[kRows] = {
            nn_block8<1>, nn_block8<2>, nn_block8<3>, nn_block8<4>, nn_block8<5>, nn_block8<6>};
    return Width == kColumns ? kWide[rows - 1] : kNarrow[rows - 1];
}

void nn(int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc, bool accumulate) {
    int j = 0;
    for (; j + kColumns <= n; j += kColumns) {
        for (int i = 0; i < m; i += kRows) {
            const int rows = std::min(kRows, m - i);
            nn_block_for<kColumns>(rows)(k, a + static_cast<std::ptrdiff_t>(i) * lda, lda, b + j, ldb,
                    c + static_cast<std::ptrdiff_t>(i) * ldc + j, ldc, accumulate);
        }
    }
    for (; j + 8 <= n; j += 8) {
        for (int i = 0; i < m; i += kRows) {
            const int rows = std::min(kRows, m - i);
            nn_block_for<8>(rows)(k, a + static_cast<std::ptrdiff_t>(i) * lda, lda, b + j, ldb,
                    c + static_cast<std::ptrdiff_t>(i) * ldc + j, ldc, accumulate);
        }
    }
    // Fewer than 8 columns left: one at a time.
    for (; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            float sum = accumulate ? c[static_cast<std::ptrdiff_t>(i) * ldc + j] : 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                sum += a[static_cast<std::ptrdiff_t>(i) * lda + kk] * b[static_cast<std::ptrdiff_t>(kk) * ldb + j];
            }
            c[static_cast<std::ptrdiff_t>(i) * ldc + j] = sum;
        }
    }
}

float horizontal_sum(__m256 value) {
    const __m128 sum4 = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    const __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    return _mm_cvtss_f32(_mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1)));
}

// R rows of A against P rows of B: R*P dot products at once, each a vector of partial sums over k, added up at the end.
// Per step of 8 values: R + P loads feed R*P multiply-adds.
template<int R, int P>
void nt_block(int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc, bool accumulate) {
    __m256 acc[R][P];
    for (int r = 0; r < R; ++r) {
        for (int p = 0; p < P; ++p) {
            acc[r][p] = _mm256_setzero_ps();
        }
    }
    int kk = 0;
    for (; kk + 8 <= k; kk += 8) {
        __m256 left[R];
        for (int r = 0; r < R; ++r) {
            left[r] = _mm256_loadu_ps(a + static_cast<std::ptrdiff_t>(r) * lda + kk);
        }
        for (int p = 0; p < P; ++p) {
            const __m256 right = _mm256_loadu_ps(b + static_cast<std::ptrdiff_t>(p) * ldb + kk);
            for (int r = 0; r < R; ++r) {
                acc[r][p] = _mm256_fmadd_ps(left[r], right, acc[r][p]);
            }
        }
    }
    for (int r = 0; r < R; ++r) {
        for (int p = 0; p < P; ++p) {
            float sum = horizontal_sum(acc[r][p]);
            for (int t = kk; t < k; ++t) {
                sum += a[static_cast<std::ptrdiff_t>(r) * lda + t] * b[static_cast<std::ptrdiff_t>(p) * ldb + t];
            }
            float& target = c[static_cast<std::ptrdiff_t>(r) * ldc + p];
            target = accumulate ? target + sum : sum;
        }
    }
}

using NtBlock = void (*)(int, const float*, int, const float*, int, float*, int, bool);

NtBlock nt_block_for(int rows, int columns) {
    static constexpr NtBlock kBlocks[2][4] = {
            {nt_block<1, 1>, nt_block<1, 2>, nt_block<1, 3>, nt_block<1, 4>},
            {nt_block<2, 1>, nt_block<2, 2>, nt_block<2, 3>, nt_block<2, 4>}};
    return kBlocks[rows - 1][columns - 1];
}

void nt(int m, int p, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc, bool accumulate) {
    for (int i = 0; i < m; i += 2) {
        const int rows = std::min(2, m - i);
        for (int j = 0; j < p; j += 4) {
            const int columns = std::min(4, p - j);
            nt_block_for(rows, columns)(k, a + static_cast<std::ptrdiff_t>(i) * lda, lda,
                    b + static_cast<std::ptrdiff_t>(j) * ldb, ldb, c + static_cast<std::ptrdiff_t>(i) * ldc + j, ldc,
                    accumulate);
        }
    }
}

}

const GemmKernels kAvx2Gemm{nn, nt, "avx2"};

}
