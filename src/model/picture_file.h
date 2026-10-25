#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace game {

// A photo from disk made into what the network takes: the same crop and shrink the normalization of a dataset does
// (kImageBytes bytes). Fails when the file cannot be read as an image.
[[nodiscard]] std::expected<std::vector<std::uint8_t>, std::string> load_normalized_picture(
        const std::filesystem::path& path);

}
