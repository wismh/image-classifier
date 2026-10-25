#pragma once

#include "model/image_decoder.h"

#include <cstdint>
#include <vector>

namespace game {

// The largest centered square of the photo, averaged down (or blown up) to kImageSide x kImageSide. The result holds
// kImageBytes bytes. An empty photo gives a black picture.
[[nodiscard]] std::vector<std::uint8_t> to_square_picture(const RgbImage& image);

}
