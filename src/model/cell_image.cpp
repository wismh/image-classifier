#include "model/cell_image.h"

#include <algorithm>

namespace game {

CellImage gray_image(std::span<const float> map, int side, float scale) {
    CellImage image;
    image.kind = CellImage::Kind::Gray;
    image.width = side;
    image.height = side;
    image.values.resize(map.size());
    for (std::size_t i = 0; i < map.size(); ++i) {
        image.values[i] = std::clamp(map[i] * scale, 0.0f, 1.0f);
    }
    return image;
}

CellImage color_image(std::span<const std::uint8_t> rgb, int side) {
    CellImage image;
    image.kind = CellImage::Kind::Color;
    image.width = side;
    image.height = side;
    image.values.resize(rgb.size());
    for (std::size_t i = 0; i < rgb.size(); ++i) {
        image.values[i] = static_cast<float>(rgb[i]) / 255.0f;
    }
    return image;
}

CellImage weights_image(const FilterPicture& picture) {
    CellImage image;
    image.kind = picture.kind == FilterPicture::Kind::Color ? CellImage::Kind::Color : CellImage::Kind::Signed;
    image.width = picture.width;
    image.height = picture.height;
    image.values = picture.values;
    return image;
}

CellImage downsampled(const CellImage& image, int max_side) {
    const int larger = std::max(image.width, image.height);
    if (image.kind == CellImage::Kind::None || larger <= max_side) {
        return image;
    }
    // The smallest whole factor that makes it fit; a size that does not divide by it loses its last rows and columns.
    const int factor = (larger + max_side - 1) / max_side;
    const int channels = image.kind == CellImage::Kind::Color ? 3 : 1;
    CellImage result;
    result.kind = image.kind;
    result.width = image.width / factor;
    result.height = image.height / factor;
    result.values.assign(static_cast<std::size_t>(result.width) * result.height * channels, 0.0f);
    for (int y = 0; y < result.height; ++y) {
        for (int x = 0; x < result.width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int dy = 0; dy < factor; ++dy) {
                    for (int dx = 0; dx < factor; ++dx) {
                        sum += image.values[(static_cast<std::size_t>(y * factor + dy) * image.width + x * factor + dx)
                                        * channels + c];
                    }
                }
                result.values[(static_cast<std::size_t>(y) * result.width + x) * channels + c] =
                        sum / static_cast<float>(factor * factor);
            }
        }
    }
    return result;
}

}
