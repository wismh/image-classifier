#pragma once

#include "model/filter_view.h"

#include <cstdint>
#include <span>
#include <vector>

namespace game {

// A small picture ready to be drawn pixel by pixel: a feature map, a filter's weights or a dream.
struct CellImage {
    enum class Kind { None, Gray, Signed, Color };

    Kind kind = Kind::None;
    int width = 0;
    int height = 0;
    // Gray: width * height values in 0..1. Signed: width * height values in -1..1. Color: width * height * 3 values in
    // 0..1.
    std::vector<float> values;
};

// A map of side x side values (a filter's answers) as a gray picture: every value times `scale`, cut to 0..1. A
// `scale` of 0 gives black.
[[nodiscard]] CellImage gray_image(std::span<const float> map, int side, float scale);
// A normalized picture of a dataset (side x side rows of RGB bytes) as a color picture.
[[nodiscard]] CellImage color_image(std::span<const std::uint8_t> rgb, int side);
// A filter's weights as they come from filter_weights_picture().
[[nodiscard]] CellImage weights_image(const FilterPicture& picture);
// The picture made smaller by a whole factor so that neither side is more than `max_side` (each new pixel is the mean
// of a
// block), or the picture itself when it is small enough already.
[[nodiscard]] CellImage downsampled(const CellImage& image, int max_side);

}
