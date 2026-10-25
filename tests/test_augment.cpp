#include "model/augment.h"

#include "model/pet_image.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace game {

namespace {

constexpr std::size_t kPlane = static_cast<std::size_t>(kImageSide) * kImageSide;

// Every pixel has its own value in the first channel, so any move shows.
std::vector<std::uint8_t> ramp() {
    std::vector<std::uint8_t> picture(kImageBytes);
    for (std::size_t i = 0; i < kPlane; ++i) {
        picture[i * kChannels] = static_cast<std::uint8_t>(i % 251);
        picture[i * kChannels + 1] = static_cast<std::uint8_t>(10);
        picture[i * kChannels + 2] = static_cast<std::uint8_t>(250);
    }
    return picture;
}

std::vector<float> plain(const std::vector<std::uint8_t>& picture) {
    std::vector<float> input(kChannels * kPlane);
    to_input(picture, input);
    return input;
}

}

TEST(ToInput, ScalesBytesToHalfRangeAndGroupsByChannel) {
    std::vector<std::uint8_t> picture(kImageBytes, 0);
    picture[0] = 255;      // pixel 0, red
    picture[1] = 51;       // pixel 0, green
    picture[2] = 0;        // pixel 0, blue
    picture[3] = 0;        // pixel 1, red
    picture[4] = 255;      // pixel 1, green
    std::vector<float> input(kChannels * kPlane);
    to_input(picture, input);
    EXPECT_FLOAT_EQ(input[0], 0.5f);
    EXPECT_FLOAT_EQ(input[kPlane], 51.0f / 255.0f - 0.5f);
    EXPECT_FLOAT_EQ(input[2 * kPlane], -0.5f);
    EXPECT_FLOAT_EQ(input[1], -0.5f);
    EXPECT_FLOAT_EQ(input[kPlane + 1], 0.5f);
}

TEST(Augmentation, NoneLeavesThePictureAsItIs) {
    const auto picture = ramp();
    std::vector<float> input(kChannels * kPlane);
    std::mt19937 rng{5};
    to_augmented_input(picture, input, Augmentation::none(), rng);
    EXPECT_EQ(input, plain(picture));
}

TEST(Augmentation, FlipMirrorsLeftToRightAndHappensAboutHalfTheTime) {
    const auto picture = ramp();
    const Augmentation flip_only{true, 0, 0.0f, 0.0f};
    const std::vector<float> original = plain(picture);
    int flipped = 0;
    for (std::uint32_t seed = 0; seed < 100; ++seed) {
        std::vector<float> input(kChannels * kPlane);
        std::mt19937 rng{seed};
        to_augmented_input(picture, input, flip_only, rng);
        if (input == original) {
            continue;
        }
        ++flipped;
        for (int y = 0; y < kImageSide; ++y) {
            for (int x = 0; x < kImageSide; ++x) {
                ASSERT_FLOAT_EQ(input[static_cast<std::size_t>(y * kImageSide + x)],
                        original[static_cast<std::size_t>(y * kImageSide + kImageSide - 1 - x)]);
            }
        }
    }
    EXPECT_GT(flipped, 25);
    EXPECT_LT(flipped, 75);
}

TEST(Augmentation, ShiftMovesByAtMostMaxShiftAndStretchesTheEdge) {
    const auto picture = ramp();
    const Augmentation shift_only{false, 3, 0.0f, 0.0f};
    const std::vector<float> original = plain(picture);
    bool moved = false;
    for (std::uint32_t seed = 0; seed < 40; ++seed) {
        std::vector<float> input(kChannels * kPlane);
        std::mt19937 rng{seed};
        to_augmented_input(picture, input, shift_only, rng);
        // Some (dx, dy) within +-3 must explain every pixel, with coordinates clamped at the edge.
        bool explained = false;
        for (int dy = -3; dy <= 3 && !explained; ++dy) {
            for (int dx = -3; dx <= 3 && !explained; ++dx) {
                bool all = true;
                for (int y = 0; y < kImageSide && all; ++y) {
                    for (int x = 0; x < kImageSide && all; ++x) {
                        const int sy = std::clamp(y + dy, 0, kImageSide - 1);
                        const int sx = std::clamp(x + dx, 0, kImageSide - 1);
                        all = input[static_cast<std::size_t>(y * kImageSide + x)]
                                == original[static_cast<std::size_t>(sy * kImageSide + sx)];
                    }
                }
                explained = all;
                moved = moved || (all && (dx != 0 || dy != 0));
            }
        }
        EXPECT_TRUE(explained) << "seed " << seed;
    }
    EXPECT_TRUE(moved);
}

TEST(Augmentation, ValuesStayInTheInputRange) {
    const auto picture = ramp();
    const Augmentation strong{true, 4, 0.5f, 0.5f};
    for (std::uint32_t seed = 0; seed < 20; ++seed) {
        std::vector<float> input(kChannels * kPlane);
        std::mt19937 rng{seed};
        to_augmented_input(picture, input, strong, rng);
        EXPECT_GE(*std::ranges::min_element(input), -0.5f);
        EXPECT_LE(*std::ranges::max_element(input), 0.5f);
    }
}

TEST(Augmentation, BrightnessAndContrastChangeTheValues) {
    const std::vector<std::uint8_t> picture(kImageBytes, 128);
    const Augmentation tone_only{false, 0, 0.2f, 0.0f};
    bool changed = false;
    for (std::uint32_t seed = 0; seed < 10; ++seed) {
        std::vector<float> input(kChannels * kPlane);
        std::mt19937 rng{seed};
        to_augmented_input(picture, input, tone_only, rng);
        changed = changed || std::abs(input[0] - (128.0f / 255.0f - 0.5f)) > 0.01f;
        EXPECT_TRUE(std::ranges::all_of(input, [&](float v) { return v == input[0]; }));  // one change for the whole picture
    }
    EXPECT_TRUE(changed);
}

TEST(SampleRng, DependsOnlyOnItsArguments) {
    auto a = sample_rng(1, 2, 3);
    auto b = sample_rng(1, 2, 3);
    EXPECT_EQ(a(), b());
    EXPECT_NE(sample_rng(1, 2, 3)(), sample_rng(1, 2, 4)());
    EXPECT_NE(sample_rng(1, 2, 3)(), sample_rng(1, 3, 3)());
    EXPECT_NE(sample_rng(1, 2, 3)(), sample_rng(2, 2, 3)());
}

}
