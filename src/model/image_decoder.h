#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace game {

// A decoded photo: `pixels` holds width * height * 3 bytes (RGB), rows top to bottom.
struct RgbImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;
};

// Reads a JPEG, PNG, BMP or GIF file (the first frame of a GIF), whatever its extension says: a dataset of photos
// often holds a few PNGs and GIFs named .jpg. Grayscale and alpha are converted to plain RGB.
[[nodiscard]] std::expected<RgbImage, std::string> decode_image(const std::filesystem::path& path);

}
