#pragma once

#include "model/labeled_images.h"

#include <expected>
#include <filesystem>
#include <string>

namespace game {

// A normalized dataset on disk, one `.cvds` file: a magic, a version, the picture side, the channels and the number of
// samples, then all the label bytes, then all the pictures (kImageBytes each).
inline constexpr const char* kDatasetExtension = ".cvds";

// Writes beside the final name and renames, so a failed write never leaves half a dataset.
[[nodiscard]] std::expected<void, std::string> write_dataset_file(const std::filesystem::path& path,
        const LabeledImages& data);
[[nodiscard]] std::expected<LabeledImages, std::string> read_dataset_file(const std::filesystem::path& path);

}
