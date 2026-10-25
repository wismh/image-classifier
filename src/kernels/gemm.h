#pragma once

namespace game {

// Matrix products on row-major float matrices with a leading dimension (the distance between rows), which is all the
// network's convolutions and dense layers are made of. Two implementations: plain loops that any CPU runs, and AVX2/FMA
// kernels used when the CPU has them. Both give the same result up to the order of additions.
struct GemmKernels {
    // C[m x n] (+)= A[m x k] * B[k x n]. Without `accumulate`, C is overwritten.
    void (*nn)(int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc,
            bool accumulate);
    // C[m x p] (+)= A[m x k] * B[p x k]^T: every value is the dot product of a row of A and a row of B, both
    // contiguous. Without `accumulate`, C is overwritten.
    void (*nt)(int m, int p, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc,
            bool accumulate);
    const char* name;
};

// Runs on every CPU.
[[nodiscard]] const GemmKernels& generic_gemm();
// The AVX2/FMA kernels, or nullptr when this build has none (not an x86 build) or this CPU cannot run them.
[[nodiscard]] const GemmKernels* avx2_gemm();
// The fastest kernels this CPU can run.
[[nodiscard]] const GemmKernels& best_gemm();

}
