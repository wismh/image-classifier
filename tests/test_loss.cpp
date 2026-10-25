#include "model/loss.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace game {

TEST(SoftmaxCrossEntropy, EqualScoresGiveLogTwoAndAHalfEach) {
    std::vector<float> d(2);
    const LossResult result = softmax_cross_entropy(std::vector<float>{3.0f, 3.0f}, 1, 1.0f, d);
    EXPECT_NEAR(result.loss, std::log(2.0f), 1e-6);
    EXPECT_NEAR(d[0], 0.5f, 1e-6);
    EXPECT_NEAR(d[1], -0.5f, 1e-6);
}

TEST(SoftmaxCrossEntropy, MatchesTheFormula) {
    std::vector<float> d(2);
    const float a = 2.0f;
    const float b = -1.0f;
    const LossResult result = softmax_cross_entropy(std::vector<float>{a, b}, 0, 1.0f, d);
    const float p0 = std::exp(a) / (std::exp(a) + std::exp(b));
    EXPECT_NEAR(result.loss, -std::log(p0), 1e-6);
    EXPECT_NEAR(d[0], p0 - 1.0f, 1e-6);
    EXPECT_NEAR(d[1], 1.0f - p0, 1e-6);
}

TEST(SoftmaxCrossEntropy, WeightScalesTheGradientButNotTheLoss) {
    std::vector<float> plain(2);
    std::vector<float> heavy(2);
    const LossResult a = softmax_cross_entropy(std::vector<float>{0.3f, 1.2f}, 0, 1.0f, plain);
    const LossResult b = softmax_cross_entropy(std::vector<float>{0.3f, 1.2f}, 0, 2.5f, heavy);
    EXPECT_FLOAT_EQ(a.loss, b.loss);
    EXPECT_NEAR(heavy[0], 2.5f * plain[0], 1e-6);
    EXPECT_NEAR(heavy[1], 2.5f * plain[1], 1e-6);
}

TEST(SoftmaxCrossEntropy, HugeScoresStayFinite) {
    std::vector<float> d(2);
    const LossResult wrong = softmax_cross_entropy(std::vector<float>{1000.0f, -1000.0f}, 1, 1.0f, d);
    EXPECT_TRUE(std::isfinite(wrong.loss));
    EXPECT_NEAR(wrong.loss, 2000.0f, 1e-3);
    EXPECT_TRUE(std::isfinite(d[0]) && std::isfinite(d[1]));
    const LossResult right = softmax_cross_entropy(std::vector<float>{1000.0f, -1000.0f}, 0, 1.0f, d);
    EXPECT_NEAR(right.loss, 0.0f, 1e-6);
}

TEST(SoftmaxCrossEntropy, PredictsTheHighestScore) {
    std::vector<float> d(2);
    EXPECT_EQ(softmax_cross_entropy(std::vector<float>{0.1f, 0.2f}, 0, 1.0f, d).predicted, 1);
    EXPECT_EQ(softmax_cross_entropy(std::vector<float>{0.9f, 0.2f}, 1, 1.0f, d).predicted, 0);
}

TEST(SoftmaxCrossEntropy, GradientsOfAllScoresAddUpToZero) {
    std::vector<float> d(2);
    softmax_cross_entropy(std::vector<float>{0.7f, -0.4f}, 1, 1.0f, d);
    EXPECT_NEAR(d[0] + d[1], 0.0f, 1e-6);
}

}
