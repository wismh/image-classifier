#include "model/augment.h"
#include "model/grad_cam.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>

namespace game {

namespace {

constexpr CnnConfig kSmall{4, 9, 3, 8};

std::vector<std::uint8_t> noisy_picture(std::uint32_t seed) {
    std::mt19937 rng{seed};
    std::vector<std::uint8_t> picture(kImageBytes);
    for (auto& byte : picture) {
        byte = static_cast<std::uint8_t>(rng() % 256);
    }
    return picture;
}

}

TEST(GradCam, MapsCoverTheLastBlockWithValuesFromZeroToOne) {
    const Cnn cnn{kSmall, 3};
    Cnn::Workspace workspace = cnn.make_workspace();
    const GradCam cam = grad_cam(cnn, workspace, noisy_picture(1));

    EXPECT_EQ(cam.side, Cnn::block_side(Cnn::kBlocks - 1));
    float strongest = 0.0f;
    for (const auto& map : cam.maps) {
        ASSERT_EQ(map.size(), static_cast<std::size_t>(cam.side) * cam.side);
        EXPECT_GE(*std::ranges::min_element(map), 0.0f);
        strongest = std::max(strongest, *std::ranges::max_element(map));
    }
    EXPECT_FLOAT_EQ(strongest, 1.0f);
}

TEST(GradCam, EachPlaceIsTheRectifiedSumOfTheMapsWeighedByTheMeanGradient) {
    const Cnn cnn{kSmall, 5};
    Cnn::Workspace workspace = cnn.make_workspace();
    const std::vector<std::uint8_t> picture = noisy_picture(2);
    const GradCam cam = grad_cam(cnn, workspace, picture);

    // The sum rebuilt by hand from the run's own answers and gradients.
    constexpr int kLast = Cnn::kBlocks - 1;
    const std::size_t plane = static_cast<std::size_t>(cam.side) * cam.side;
    std::array<std::vector<float>, kNumClasses> expected_maps;
    float strongest = 0.0f;
    for (std::size_t species = 0; species < kNumClasses; ++species) {
        std::vector<float> input(Cnn::kInputSize);
        to_input(picture, input);
        (void)cnn.forward(input, workspace);
        std::array<float, kNumClasses> d_scores{};
        d_scores.fill(-1.0f);
        d_scores[species] = 1.0f;
        std::vector<float> unused(cnn.param_count(), 0.0f);
        cnn.backward(input, workspace, d_scores, unused);

        std::vector<float> expected(plane, 0.0f);
        for (int filter = 0; filter < cnn.filters(kLast); ++filter) {
            const std::size_t first = static_cast<std::size_t>(filter) * plane;
            double mean = 0.0;
            for (std::size_t i = 0; i < plane; ++i) {
                mean += workspace.deltas[Cnn::relu_layer(kLast) + 1][first + i];
            }
            mean /= static_cast<double>(plane);
            for (std::size_t i = 0; i < plane; ++i) {
                expected[i] += static_cast<float>(mean) * workspace.outputs[Cnn::relu_layer(kLast)][first + i];
            }
        }
        for (float& value : expected) {
            value = std::max(value, 0.0f);
            strongest = std::max(strongest, value);
        }
        expected_maps[species] = std::move(expected);
    }
    ASSERT_GT(strongest, 0.0f);
    for (std::size_t species = 0; species < kNumClasses; ++species) {
        for (std::size_t i = 0; i < plane; ++i) {
            EXPECT_NEAR(cam.maps[species][i], expected_maps[species][i] / strongest, 1e-5f)
                    << "species " << species << " place " << i;
        }
    }
}

TEST(GradCam, APlaceArguesForOneSpeciesOrTheOtherNeverBoth) {
    const Cnn cnn{kSmall, 13};
    Cnn::Workspace workspace = cnn.make_workspace();
    const GradCam cam = grad_cam(cnn, workspace, noisy_picture(6));
    const auto& cat = cam.maps[static_cast<std::size_t>(ClassId::First)];
    const auto& dog = cam.maps[static_cast<std::size_t>(ClassId::Second)];
    for (std::size_t i = 0; i < cat.size(); ++i) {
        EXPECT_TRUE(cat[i] == 0.0f || dog[i] == 0.0f) << "place " << i;
    }
}

TEST(GradCam, TheTwoSpeciesDoNotGetTheSameMap) {
    const Cnn cnn{kSmall, 7};
    Cnn::Workspace workspace = cnn.make_workspace();
    const GradCam cam = grad_cam(cnn, workspace, noisy_picture(3));
    EXPECT_NE(cam.maps[static_cast<std::size_t>(ClassId::First)], cam.maps[static_cast<std::size_t>(ClassId::Second)]);
}

TEST(GradCam, ANetworkWithNoWeightsFindsNoEvidence) {
    Cnn cnn{kSmall, 9};
    std::ranges::fill(cnn.params(), 0.0f);
    Cnn::Workspace workspace = cnn.make_workspace();
    const GradCam cam = grad_cam(cnn, workspace, noisy_picture(4));
    for (const auto& map : cam.maps) {
        EXPECT_EQ(*std::ranges::max_element(map), 0.0f);
    }
}

TEST(GradCam, TheSamePictureGivesTheSameMaps) {
    const Cnn cnn{kSmall, 11};
    Cnn::Workspace first = cnn.make_workspace();
    Cnn::Workspace second = cnn.make_workspace();
    const std::vector<std::uint8_t> picture = noisy_picture(5);
    EXPECT_EQ(grad_cam(cnn, first, picture).maps, grad_cam(cnn, second, picture).maps);
}

}
