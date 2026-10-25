#include "model/cnn.h"

#include "model/loss.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace game {

namespace {

// Small enough to check every parameter against a numerical gradient in a moment.
constexpr CnnConfig kSmall{2, 3, 2, 5};

std::vector<float> random_picture(std::uint32_t seed) {
    std::mt19937 rng{seed};
    std::uniform_real_distribution<float> uniform{-0.5f, 0.5f};
    std::vector<float> picture(Cnn::kInputSize);
    for (float& value : picture) {
        value = uniform(rng);
    }
    return picture;
}

double loss_at(const Cnn& cnn, const std::vector<float>& picture, int label) {
    Cnn::Workspace workspace = cnn.make_workspace();
    std::vector<float> d(kNumClasses);
    return softmax_cross_entropy(cnn.forward(picture, workspace), label, 1.0f, d).loss;
}

}

TEST(CnnConfig, ValidatesTheSizes) {
    EXPECT_TRUE(CnnConfig{}.valid());
    EXPECT_TRUE((CnnConfig{1, 1, 1, 1}.valid()));
    EXPECT_FALSE((CnnConfig{0, 16, 32, 64}.valid()));
    EXPECT_FALSE((CnnConfig{8, 16, 65, 64}.valid()));
    EXPECT_FALSE((CnnConfig{8, 16, 32, 0}.valid()));
    EXPECT_FALSE((CnnConfig{8, 16, 32, 513}.valid()));
    EXPECT_EQ((CnnConfig{8, 16, 32, 64}.label()), "8-16-32-64");
}

TEST(Cnn, ParameterCountIsTheSumOfItsLayers) {
    const Cnn cnn{CnnConfig{8, 16, 32, 64}, 1};
    const std::size_t conv1 = 8 * 3 * 9 + 8;
    const std::size_t conv2 = 16 * 8 * 9 + 16;
    const std::size_t conv3 = 32 * 16 * 9 + 32;
    const std::size_t dense1 = 32 * 8 * 8 * 64 + 64;
    const std::size_t dense2 = 64 * 2 + 2;
    EXPECT_EQ(cnn.param_count(), conv1 + conv2 + conv3 + dense1 + dense2);
}

TEST(Cnn, SameSeedSameWeightsOtherSeedOtherWeights) {
    const Cnn a{kSmall, 7};
    const Cnn b{kSmall, 7};
    const Cnn c{kSmall, 8};
    EXPECT_TRUE(std::ranges::equal(a.params(), b.params()));
    EXPECT_FALSE(std::ranges::equal(a.params(), c.params()));
}

TEST(Cnn, ForwardGivesOneScorePerClassAndIsRepeatable) {
    const Cnn cnn{kSmall, 3};
    const std::vector<float> picture = random_picture(1);
    Cnn::Workspace workspace = cnn.make_workspace();
    const auto first = cnn.forward(picture, workspace);
    ASSERT_EQ(first.size(), kNumClasses);
    const std::vector<float> scores(first.begin(), first.end());
    const auto second = cnn.forward(picture, workspace);
    EXPECT_TRUE(std::ranges::equal(scores, second));
    EXPECT_TRUE(std::ranges::all_of(scores, [](float s) { return std::isfinite(s); }));
}

TEST(Cnn, FreshNetworkIsNotWildlyConfident) {
    // He initialization keeps the scores of a fresh network at a sensible size.
    const Cnn cnn{CnnConfig{}, 5};
    Cnn::Workspace workspace = cnn.make_workspace();
    for (std::uint32_t seed = 0; seed < 5; ++seed) {
        for (const float score : cnn.forward(random_picture(seed), workspace)) {
            EXPECT_LT(std::abs(score), 20.0f);
        }
    }
}

TEST(Cnn, WithParamsRestoresTheSameNetwork) {
    const Cnn original{kSmall, 11};
    const std::vector<float> picture = random_picture(2);
    const std::vector<float> params(original.params().begin(), original.params().end());
    auto restored = Cnn::with_params(kSmall, params);
    ASSERT_TRUE(restored) << restored.error();

    Cnn::Workspace a = original.make_workspace();
    Cnn::Workspace b = restored->make_workspace();
    EXPECT_TRUE(std::ranges::equal(original.forward(picture, a), restored->forward(picture, b)));
}

TEST(Cnn, WithParamsRejectsAWrongSizeOrAnInvalidConfig) {
    const Cnn original{kSmall, 11};
    std::vector<float> params(original.params().begin(), original.params().end());
    params.pop_back();
    EXPECT_FALSE(Cnn::with_params(kSmall, params));
    EXPECT_FALSE(Cnn::with_params(CnnConfig{0, 1, 1, 1}, {}));
    EXPECT_FALSE(Cnn::with_params(CnnConfig{2, 3, 2, 6}, std::vector<float>(original.param_count())));
}

TEST(Cnn, GradientsOfTheWholeNetworkMatchNumericalOnes) {
    Cnn cnn{kSmall, 21};
    const std::vector<float> picture = random_picture(4);
    constexpr int kLabel = 1;

    Cnn::Workspace workspace = cnn.make_workspace();
    std::vector<float> d_scores(kNumClasses);
    softmax_cross_entropy(cnn.forward(picture, workspace), kLabel, 1.0f, d_scores);
    std::vector<float> grads(cnn.param_count(), 0.0f);
    cnn.backward(picture, workspace, d_scores, grads);

    // Every parameter, so a bug in any one layer's slice of the vector shows.
    constexpr float kStep = 3e-4f;
    int checked = 0;
    int off = 0;
    for (std::size_t i = 0; i < cnn.param_count(); ++i) {
        const float original = cnn.params()[i];
        cnn.params()[i] = original + kStep;
        const double up = loss_at(cnn, picture, kLabel);
        cnn.params()[i] = original - kStep;
        const double down = loss_at(cnn, picture, kLabel);
        cnn.params()[i] = original;
        const double numeric = (up - down) / (2.0 * kStep);
        // A step that crosses a ReLU or max-pool kink makes the numerical value unreliable for that one parameter: the
        // first convolution's weights touch thousands of activations, so a few of them sit within a step of a kink.
        const bool close = std::abs(grads[i] - numeric) <= 2e-2 * std::max(0.05, std::abs(numeric));
        off += close ? 0 : 1;
        ++checked;
    }
    EXPECT_GT(checked, 300);
    EXPECT_LE(off, checked / 50) << off << " of " << checked << " parameters disagree";
}

TEST(Cnn, TrainingStepOnOneSampleLowersItsLoss) {
    Cnn cnn{kSmall, 31};
    const std::vector<float> picture = random_picture(6);
    constexpr int kLabel = 0;
    const double before = loss_at(cnn, picture, kLabel);

    Cnn::Workspace workspace = cnn.make_workspace();
    std::vector<float> d_scores(kNumClasses);
    softmax_cross_entropy(cnn.forward(picture, workspace), kLabel, 1.0f, d_scores);
    std::vector<float> grads(cnn.param_count(), 0.0f);
    cnn.backward(picture, workspace, d_scores, grads);
    for (std::size_t i = 0; i < grads.size(); ++i) {
        cnn.params()[i] -= 0.01f * grads[i];
    }
    EXPECT_LT(loss_at(cnn, picture, kLabel), before);
}

TEST(Cnn, WorkspacesOfOneNetworkDoNotShareState) {
    const Cnn cnn{kSmall, 3};
    const std::vector<float> one = random_picture(1);
    const std::vector<float> two = random_picture(2);
    Cnn::Workspace a = cnn.make_workspace();
    Cnn::Workspace b = cnn.make_workspace();
    const auto scores_a = cnn.forward(one, a);
    const std::vector<float> kept(scores_a.begin(), scores_a.end());
    (void)cnn.forward(two, b);
    EXPECT_TRUE(std::ranges::equal(kept, scores_a));
}

}
