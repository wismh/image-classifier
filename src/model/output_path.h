#pragma once

#include <filesystem>

namespace game {

// The file a normalized dataset is written to: in `output_directory`, or beside the photos folder when that is empty,
// named like the photos folder. Empty while no photos folder is chosen.
[[nodiscard]] std::filesystem::path planned_output_path(const std::filesystem::path& source,
        const std::filesystem::path& output_directory);

}
