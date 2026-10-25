#include "model/dream.h"

#include "model/loss.h"
#include "model/prediction.h"
#include "model/training_run.h"

#include "test_support.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <random>

namespace game {

namespace {

constexpr CnnConfig kSmall{2, 2, 2, 8};

// A network that tells dark pictures (cats) from bright ones (dogs).
Cnn trained_network() {
    TrainingConfig config;
    config.epochs = 6;
    config.batch_size = 8;
    config.learning_rate = 5e-3f;
    config.augment = false;
    TrainingRun run{Cnn{kSmall, 1}, std::make_shared<const LabeledImages>(test::synthetic_set(80, 1)), config,
            Parallel{}};
    while (!run.step(std::chrono::milliseconds{50})) {
    }
    return run.take_model();
}

}

TEST(CnnInputGradient, MatchesNumericalOnes) {
    const Cnn cnn{kSmall, 21};
    std::mt19937 rng{4};
    std::uniform_real_distribution<float> uniform{-0.5f, 0.5f};
    std::vector<float> picture(Cnn::kInputSize);
    for (float& value : picture) {
        value = uniform(rng);
    }
    constexpr int kLabel = 1;

    Cnn::Workspace workspace = cnn.make_workspace(true);
    std::vector<float> d_scores(kNumClasses);
    softmax_cross_entropy(cnn.forward(picture, workspace), kLabel, 1.0f, d_scores);
    std::vector<float> grads(cnn.param_count(), 0.0f);
    cnn.backward(picture, workspace, d_scores, grads);
    const std::vector<float> analytic(workspace.input_gradient().begin(), workspace.input_gradient().end());
    ASSERT_EQ(analytic.size(), Cnn::kInputSize);

    const auto loss_at = [&](const std::vector<float>& input) {
        Cnn::Workspace plain = cnn.make_workspace();
        std::vector<float> d(kNumClasses);
        return static_cast<double>(softmax_cross_entropy(cnn.forward(input, plain), kLabel, 1.0f, d).loss);
    };
    constexpr float kStep = 3e-4f;
    int checked = 0;
    int off = 0;
    for (std::size_t i = 0; i < picture.size(); i += 97) {
        std::vector<float> moved = picture;
        moved[i] += kStep;
        const double up = loss_at(moved);
        moved[i] -= 2.0f * kStep;
        const double down = loss_at(moved);
        const double numeric = (up - down) / (2.0 * kStep);
        off += std::abs(analytic[i] - numeric) <= 2e-2 * std::max(0.05, std::abs(numeric)) ? 0 : 1;
        ++checked;
    }
    EXPECT_GT(checked, 100);
    EXPECT_LE(off, checked / 20) << off << " of " << checked << " pixels disagree";
}

TEST(CnnInputGradient, IsNotComputedUnlessAskedFor) {
    const Cnn cnn{kSmall, 1};
    EXPECT_TRUE(cnn.make_workspace().deltas.front().empty());
    EXPECT_EQ(cnn.make_workspace(true).deltas.front().size(), Cnn::kInputSize);
}

TEST(Dream, MakesAPictureTheNetworkTakesForTheSpecies) {
    const Cnn model = trained_network();
    const std::atomic<bool> cancel{false};
    Cnn::Workspace workspace = model.make_workspace();
    for (const ClassId species : {ClassId::First, ClassId::Second}) {
        const auto picture = dream_class(model, species, cancel);
        ASSERT_TRUE(picture);
        ASSERT_EQ(picture->size(), kImageBytes);
        const Prediction prediction = predict(model, workspace, *picture);
        EXPECT_EQ(prediction.chosen(), species);
        EXPECT_GT(prediction.confidence(), 0.9f) << "the ideal picture should be one the network is sure about";
    }
}

TEST(Dream, TheTwoSpeciesGetDifferentPictures) {
    const Cnn model = trained_network();
    const std::atomic<bool> cancel{false};
    EXPECT_NE(dream_class(model, ClassId::First, cancel), dream_class(model, ClassId::Second, cancel));
}

TEST(Dream, GivesTheSamePictureEveryTime) {
    const Cnn model = trained_network();
    const std::atomic<bool> cancel{false};
    EXPECT_EQ(dream_class(model, ClassId::Second, cancel), dream_class(model, ClassId::Second, cancel));
}

TEST(Dream, StopsWhenCancelled) {
    const Cnn model = trained_network();
    const std::atomic<bool> cancel{true};
    EXPECT_FALSE(dream_class(model, ClassId::First, cancel));
}

}
