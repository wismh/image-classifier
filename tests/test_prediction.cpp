#include "model/picture_file.h"
#include "model/prediction.h"
#include "model/training_run.h"

#include "test_support.h"

#include <gtest/gtest.h>

#include <chrono>

namespace game {

namespace {

constexpr CnnConfig kSmall{2, 2, 2, 8};

std::vector<std::uint8_t> flat_picture(std::uint8_t value) {
    return std::vector<std::uint8_t>(kImageBytes, value);
}

}

TEST(Prediction, ProbabilitiesAreSensibleAndAddUpToOne) {
    const Cnn cnn{kSmall, 1};
    Cnn::Workspace workspace = cnn.make_workspace();
    for (const std::uint8_t value : {0, 100, 255}) {
        const Prediction p = predict(cnn, workspace, flat_picture(value));
        EXPECT_GE(p.first, 0.0f);
        EXPECT_LE(p.first, 1.0f);
        EXPECT_NEAR(p.first + p.second, 1.0f, 1e-6);
        EXPECT_FLOAT_EQ(p.confidence(), std::max(p.first, p.second));
    }
}

TEST(Prediction, SpeciesIsTheMoreLikelyOne) {
    EXPECT_EQ((Prediction{0.8f, 0.2f}.chosen()), ClassId::First);
    EXPECT_EQ((Prediction{0.3f, 0.7f}.chosen()), ClassId::Second);
    EXPECT_FLOAT_EQ((Prediction{0.3f, 0.7f}.confidence()), 0.7f);
}

TEST(Prediction, MatchesTheScoresOfTheNetwork) {
    const Cnn cnn{kSmall, 2};
    Cnn::Workspace workspace = cnn.make_workspace();
    const auto picture = flat_picture(90);
    const Prediction p = predict(cnn, workspace, picture);

    std::vector<float> input(Cnn::kInputSize);
    to_input(picture, input);
    Cnn::Workspace other = cnn.make_workspace();
    const auto scores = cnn.forward(input, other);
    EXPECT_EQ(p.chosen() == ClassId::Second, scores[1] > scores[0]);
    EXPECT_NEAR(p.second, 1.0f / (1.0f + std::exp(scores[0] - scores[1])), 1e-5);
}

TEST(Prediction, ATrainedNetworkTellsTheSpeciesApart) {
    TrainingConfig config;
    config.epochs = 5;
    config.batch_size = 8;
    config.learning_rate = 5e-3f;
    config.augment = false;
    TrainingRun run{Cnn{kSmall, 1}, std::make_shared<const LabeledImages>(test::synthetic_set(80, 1)), config,
            Parallel{}};
    while (!run.step(std::chrono::milliseconds{50})) {
    }
    const Cnn model = run.take_model();
    Cnn::Workspace workspace = model.make_workspace();

    EXPECT_EQ(predict(model, workspace, flat_picture(50)).chosen(), ClassId::First);
    EXPECT_EQ(predict(model, workspace, flat_picture(210)).chosen(), ClassId::Second);
    EXPECT_GT(predict(model, workspace, flat_picture(210)).second, predict(model, workspace, flat_picture(50)).second);
}

TEST(PictureFile, GivesANormalizedPictureOfAnyPhotoSize) {
    const test::TempDir dir;
    test::write_bmp(dir / "wide.bmp", 90, 40, test::Rgb{200, 100, 50});
    const auto picture = load_normalized_picture(dir / "wide.bmp");
    ASSERT_TRUE(picture) << picture.error();
    ASSERT_EQ(picture->size(), kImageBytes);
    EXPECT_EQ((*picture)[0], 200);
    EXPECT_EQ((*picture)[1], 100);
    EXPECT_EQ((*picture)[2], 50);
}

TEST(PictureFile, RejectsAFileThatIsNotAnImage) {
    const test::TempDir dir;
    test::write_text(dir / "note.jpg", "not a picture");
    EXPECT_FALSE(load_normalized_picture(dir / "note.jpg"));
    EXPECT_FALSE(load_normalized_picture(dir / "missing.jpg"));
}

}
