#include "model/image_decoder.h"
#include "model/image_resize.h"

#include "test_support.h"

#include <gtest/gtest.h>

namespace game {

namespace {

RgbImage solid(int width, int height, test::Rgb rgb) {
    RgbImage image{width, height, {}};
    for (int i = 0; i < width * height; ++i) {
        image.pixels.insert(image.pixels.end(), rgb.begin(), rgb.end());
    }
    return image;
}

std::uint8_t at(const std::vector<std::uint8_t>& picture, int x, int y, int channel) {
    return picture[(static_cast<std::size_t>(y) * kImageSide + x) * kChannels + channel];
}

}

TEST(ImageResize, ResultHasTheNormalizedSize) {
    EXPECT_EQ(to_square_picture(solid(500, 375, {10, 20, 30})).size(), kImageBytes);
    EXPECT_EQ(to_square_picture(solid(30, 40, {10, 20, 30})).size(), kImageBytes);
}

TEST(ImageResize, SolidColorStaysSolid) {
    for (const auto& [width, height] : {std::pair{500, 375}, std::pair{245, 279}, std::pair{64, 64}, std::pair{20, 20}}) {
        const auto picture = to_square_picture(solid(width, height, {200, 100, 7}));
        for (int y = 0; y < kImageSide; ++y) {
            for (int x = 0; x < kImageSide; ++x) {
                ASSERT_EQ(at(picture, x, y, 0), 200) << width << "x" << height;
                ASSERT_EQ(at(picture, x, y, 1), 100);
                ASSERT_EQ(at(picture, x, y, 2), 7);
            }
        }
    }
}

TEST(ImageResize, KeepsTheCenterSquareOfAWidePhoto) {
    // 300 x 100: red outside the middle 100 x 100 square, green inside it.
    RgbImage image{300, 100, {}};
    for (int y = 0; y < 100; ++y) {
        for (int x = 0; x < 300; ++x) {
            const bool middle = x >= 100 && x < 200;
            const test::Rgb rgb = middle ? test::Rgb{0, 255, 0} : test::Rgb{255, 0, 0};
            image.pixels.insert(image.pixels.end(), rgb.begin(), rgb.end());
        }
    }
    const auto picture = to_square_picture(image);
    EXPECT_EQ(at(picture, 0, 0, 1), 255);
    EXPECT_EQ(at(picture, kImageSide - 1, kImageSide - 1, 1), 255);
    EXPECT_EQ(at(picture, 0, 0, 0), 0);
}

TEST(ImageResize, KeepsTheCenterSquareOfATallPhoto) {
    RgbImage image{100, 300, {}};
    for (int y = 0; y < 300; ++y) {
        for (int x = 0; x < 100; ++x) {
            const bool middle = y >= 100 && y < 200;
            const test::Rgb rgb = middle ? test::Rgb{0, 0, 255} : test::Rgb{255, 255, 255};
            image.pixels.insert(image.pixels.end(), rgb.begin(), rgb.end());
        }
    }
    const auto picture = to_square_picture(image);
    EXPECT_EQ(at(picture, 5, 5, 2), 255);
    EXPECT_EQ(at(picture, 5, 5, 0), 0);
    EXPECT_EQ(at(picture, 60, 60, 0), 0);
}

TEST(ImageResize, AveragesAHalfAndHalfEdge) {
    // 128 x 128, left half black, right half white: the two center columns are the edge, the sides stay pure.
    RgbImage image{128, 128, {}};
    for (int y = 0; y < 128; ++y) {
        for (int x = 0; x < 128; ++x) {
            const std::uint8_t value = x < 64 ? 0 : 255;
            image.pixels.insert(image.pixels.end(), {value, value, value});
        }
    }
    const auto picture = to_square_picture(image);
    EXPECT_EQ(at(picture, 0, 10, 0), 0);
    EXPECT_EQ(at(picture, 31, 10, 0), 0);
    EXPECT_EQ(at(picture, 32, 10, 0), 255);
    EXPECT_EQ(at(picture, 63, 10, 0), 255);
}

TEST(ImageResize, EmptyPhotoGivesBlack) {
    const auto picture = to_square_picture(RgbImage{});
    ASSERT_EQ(picture.size(), kImageBytes);
    EXPECT_EQ(std::ranges::count(picture, 0), static_cast<std::ptrdiff_t>(kImageBytes));
}

TEST(ImageDecoder, ReadsABmpAndKeepsTheColors) {
    const test::TempDir dir;
    test::write_bmp(dir / "a.bmp", 5, 3, [](int x, int y) {
        return test::Rgb{static_cast<std::uint8_t>(x * 40), static_cast<std::uint8_t>(y * 80), 9};
    });
    const auto image = decode_image(dir / "a.bmp");
    ASSERT_TRUE(image) << image.error();
    EXPECT_EQ(image->width, 5);
    EXPECT_EQ(image->height, 3);
    const std::size_t pixel = (2 * 5 + 3) * 3;  // x = 3, y = 2
    EXPECT_EQ(image->pixels[pixel], 120);
    EXPECT_EQ(image->pixels[pixel + 1], 160);
    EXPECT_EQ(image->pixels[pixel + 2], 9);
}

TEST(ImageDecoder, RejectsMissingEmptyAndBrokenFiles) {
    const test::TempDir dir;
    test::write_text(dir / "empty.jpg", "");
    test::write_text(dir / "text.jpg", "this is not a picture");
    EXPECT_FALSE(decode_image(dir / "missing.jpg"));
    EXPECT_FALSE(decode_image(dir / "empty.jpg"));
    EXPECT_FALSE(decode_image(dir / "text.jpg"));
}

}
