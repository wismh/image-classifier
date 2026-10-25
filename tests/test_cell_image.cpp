#include "model/cell_image.h"

#include <gtest/gtest.h>

namespace game {

TEST(CellImage, GrayScalesAndClampsTheMap) {
    const CellImage image = gray_image(std::vector<float>{0.0f, 0.5f, 1.0f, 4.0f}, 2, 0.5f);
    EXPECT_EQ(image.kind, CellImage::Kind::Gray);
    EXPECT_EQ(image.width, 2);
    EXPECT_EQ(image.height, 2);
    EXPECT_EQ(image.values, (std::vector<float>{0.0f, 0.25f, 0.5f, 1.0f}));
    EXPECT_EQ(gray_image(std::vector<float>{3.0f}, 1, 0.0f).values, (std::vector<float>{0.0f}));
}

TEST(CellImage, ColorFromBytesIsInZeroToOne) {
    const CellImage image = color_image(std::vector<std::uint8_t>{255, 0, 51, 0, 255, 0, 0, 0, 255, 255, 255, 255}, 2);
    EXPECT_EQ(image.kind, CellImage::Kind::Color);
    ASSERT_EQ(image.values.size(), 12u);
    EXPECT_FLOAT_EQ(image.values[0], 1.0f);
    EXPECT_FLOAT_EQ(image.values[2], 0.2f);
}

TEST(CellImage, WeightsKeepTheKindOfTheFilterPicture) {
    FilterPicture colored;
    colored.kind = FilterPicture::Kind::Color;
    colored.width = 3;
    colored.height = 3;
    colored.values.assign(27, 0.5f);
    EXPECT_EQ(weights_image(colored).kind, CellImage::Kind::Color);
    FilterPicture signed_weights;
    signed_weights.kind = FilterPicture::Kind::Signed;
    signed_weights.width = 7;
    signed_weights.height = 7;
    signed_weights.values.assign(49, 0.0f);
    const CellImage image = weights_image(signed_weights);
    EXPECT_EQ(image.kind, CellImage::Kind::Signed);
    EXPECT_EQ(image.width, 7);
}

TEST(CellImage, DownsamplingAveragesBlocks) {
    CellImage image;
    image.kind = CellImage::Kind::Gray;
    image.width = 4;
    image.height = 4;
    image.values = {0, 0, 1, 1,
                    0, 0, 1, 1,
                    1, 0, 0, 0,
                    1, 0, 0, 0};
    const CellImage small = downsampled(image, 2);
    EXPECT_EQ(small.width, 2);
    EXPECT_EQ(small.height, 2);
    EXPECT_EQ(small.values, (std::vector<float>{0.0f, 1.0f, 0.5f, 0.0f}));
}

TEST(CellImage, DownsamplingKeepsColorChannelsApart) {
    CellImage image;
    image.kind = CellImage::Kind::Color;
    image.width = 2;
    image.height = 2;
    image.values = {1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1};  // red, red, green, magenta
    const CellImage small = downsampled(image, 1);
    ASSERT_EQ(small.values.size(), 3u);
    EXPECT_FLOAT_EQ(small.values[0], 0.75f);  // red is in three of the four
    EXPECT_FLOAT_EQ(small.values[1], 0.25f);  // green in one
    EXPECT_FLOAT_EQ(small.values[2], 0.25f);  // blue in one
}

TEST(CellImage, SmallPicturesAndEmptyOnesAreLeftAlone) {
    CellImage image;
    image.kind = CellImage::Kind::Gray;
    image.width = 3;
    image.height = 3;
    image.values.assign(9, 0.5f);
    EXPECT_EQ(downsampled(image, 3).values, image.values);
    EXPECT_EQ(downsampled(image, 8).width, 3);
    EXPECT_EQ(downsampled(CellImage{}, 1).kind, CellImage::Kind::None);
}

TEST(CellImage, ASizeThatDoesNotDivideLosesItsLastRowsAndColumns) {
    CellImage image;
    image.kind = CellImage::Kind::Gray;
    image.width = 7;
    image.height = 7;
    image.values.assign(49, 1.0f);
    const CellImage small = downsampled(image, 3);  // factor 3: 7 / 3 = 2 new pixels a side
    EXPECT_EQ(small.width, 2);
    EXPECT_EQ(small.height, 2);
}

}
