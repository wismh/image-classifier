#include "kernels/gemm.h"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <string>
#include <vector>

namespace game {

namespace {

constexpr float kSentinel = 12345.0f;

struct Matrix {
    int rows;
    int columns;
    int ld;  // the distance between rows: wider than `columns`, so a kernel that ignores it shows
    std::vector<float> values;

    Matrix(int rows_, int columns_, std::mt19937& rng, bool fill = true)
        : rows(rows_)
        , columns(columns_)
        , ld(columns_ + 3)
        , values(static_cast<std::size_t>(rows_) * (columns_ + 3), kSentinel) {
        std::uniform_real_distribution<float> uniform{-1.0f, 1.0f};
        for (int r = 0; r < rows && fill; ++r) {
            for (int c = 0; c < columns; ++c) {
                at(r, c) = uniform(rng);
            }
        }
    }
    float& at(int r, int c) { return values[static_cast<std::size_t>(r) * ld + c]; }
    [[nodiscard]] float at(int r, int c) const { return values[static_cast<std::size_t>(r) * ld + c]; }
};

// Everything outside the matrix proper (the padding of every row) is still the sentinel.
void expect_padding_untouched(const Matrix& m) {
    for (int r = 0; r < m.rows; ++r) {
        for (int c = m.columns; c < m.ld; ++c) {
            ASSERT_EQ(m.at(r, c), kSentinel) << "row " << r << " column " << c;
        }
    }
}

std::vector<const GemmKernels*> kernel_sets() {
    std::vector<const GemmKernels*> sets{&generic_gemm()};
    if (avx2_gemm() != nullptr) {
        sets.push_back(avx2_gemm());
    }
    return sets;
}

double tolerance(int k) {
    return 1e-5 * std::sqrt(static_cast<double>(k)) + 1e-6;
}

}

TEST(Gemm, TheFastKernelsAreChosenWhenTheCpuHasThem) {
    EXPECT_STREQ(best_gemm().name, avx2_gemm() != nullptr ? "avx2" : "generic");
}

TEST(Gemm, NnMatchesAPlainProductOnAwkwardSizes) {
    const int shapes[][3] = {{1, 1, 1}, {2, 3, 4}, {3, 5, 7}, {6, 16, 9}, {7, 17, 13}, {13, 24, 100}, {5, 8, 3},
            {1, 2048, 64}, {8, 4096, 27}, {27, 256, 32}, {144, 40, 32}, {6, 15, 1}};
    for (const GemmKernels* kernels : kernel_sets()) {
        for (const auto& shape : shapes) {
            const int m = shape[0];
            const int n = shape[1];
            const int k = shape[2];
            std::mt19937 rng{static_cast<unsigned>(m * 31 + n * 7 + k)};
            const Matrix a{m, k, rng};
            const Matrix b{k, n, rng};
            Matrix c{m, n, rng, false};
            kernels->nn(m, n, k, a.values.data(), a.ld, b.values.data(), b.ld, c.values.data(), c.ld, false);
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    double expected = 0.0;
                    for (int t = 0; t < k; ++t) {
                        expected += static_cast<double>(a.at(i, t)) * b.at(t, j);
                    }
                    ASSERT_NEAR(c.at(i, j), expected, tolerance(k))
                            << kernels->name << " " << m << "x" << n << "x" << k << " at " << i << "," << j;
                }
            }
            expect_padding_untouched(c);
        }
    }
}

TEST(Gemm, NnAccumulatesOntoWhatIsThereOrOverwrites) {
    for (const GemmKernels* kernels : kernel_sets()) {
        std::mt19937 rng{3};
        const int m = 7;
        const int n = 21;
        const int k = 11;
        const Matrix a{m, k, rng};
        const Matrix b{k, n, rng};
        Matrix start{m, n, rng};
        Matrix added = start;
        Matrix overwritten = start;
        kernels->nn(m, n, k, a.values.data(), a.ld, b.values.data(), b.ld, added.values.data(), added.ld, true);
        kernels->nn(m, n, k, a.values.data(), a.ld, b.values.data(), b.ld, overwritten.values.data(), overwritten.ld,
                false);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                ASSERT_NEAR(added.at(i, j), start.at(i, j) + overwritten.at(i, j), 1e-5) << kernels->name;
            }
        }
    }
}

TEST(Gemm, NtMatchesAPlainProductOfRowsWithRows) {
    const int shapes[][3] = {{1, 1, 1}, {2, 4, 8}, {3, 5, 7}, {5, 9, 17}, {1, 64, 2048}, {8, 27, 4096}, {32, 144, 256},
            {2, 3, 9}, {7, 2, 33}, {1, 128, 64}};
    for (const GemmKernels* kernels : kernel_sets()) {
        for (const auto& shape : shapes) {
            const int m = shape[0];
            const int p = shape[1];
            const int k = shape[2];
            std::mt19937 rng{static_cast<unsigned>(m * 13 + p * 5 + k)};
            const Matrix a{m, k, rng};
            const Matrix b{p, k, rng};
            Matrix c{m, p, rng, false};
            kernels->nt(m, p, k, a.values.data(), a.ld, b.values.data(), b.ld, c.values.data(), c.ld, false);
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < p; ++j) {
                    double expected = 0.0;
                    for (int t = 0; t < k; ++t) {
                        expected += static_cast<double>(a.at(i, t)) * b.at(j, t);
                    }
                    ASSERT_NEAR(c.at(i, j), expected, tolerance(k))
                            << kernels->name << " " << m << "x" << p << "x" << k << " at " << i << "," << j;
                }
            }
            expect_padding_untouched(c);
        }
    }
}

TEST(Gemm, NtAccumulatesOntoWhatIsThere) {
    for (const GemmKernels* kernels : kernel_sets()) {
        std::mt19937 rng{4};
        const int m = 3;
        const int p = 10;
        const int k = 37;
        const Matrix a{m, k, rng};
        const Matrix b{p, k, rng};
        Matrix start{m, p, rng};
        Matrix added = start;
        Matrix overwritten = start;
        kernels->nt(m, p, k, a.values.data(), a.ld, b.values.data(), b.ld, added.values.data(), added.ld, true);
        kernels->nt(m, p, k, a.values.data(), a.ld, b.values.data(), b.ld, overwritten.values.data(), overwritten.ld,
                false);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < p; ++j) {
                ASSERT_NEAR(added.at(i, j), start.at(i, j) + overwritten.at(i, j), 1e-5) << kernels->name;
            }
        }
    }
}

TEST(Gemm, TheKernelsAgreeWithEachOther) {
    if (avx2_gemm() == nullptr) {
        GTEST_SKIP() << "no AVX2/FMA on this CPU";
    }
    std::mt19937 rng{9};
    const Matrix a{16, 72, rng};
    const Matrix b{72, 1024, rng};
    Matrix plain{16, 1024, rng, false};
    Matrix fast{16, 1024, rng, false};
    generic_gemm().nn(16, 1024, 72, a.values.data(), a.ld, b.values.data(), b.ld, plain.values.data(), plain.ld, false);
    avx2_gemm()->nn(16, 1024, 72, a.values.data(), a.ld, b.values.data(), b.ld, fast.values.data(), fast.ld, false);
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 1024; ++j) {
            ASSERT_NEAR(plain.at(i, j), fast.at(i, j), 1e-4);
        }
    }
}

}
