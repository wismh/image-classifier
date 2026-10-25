#pragma once

#include <filesystem>
#include <string>

namespace game {

// A path short enough for a label: the whole path when it fits in `max_chars` characters, otherwise "..." and the end
// of
// it, cut at a separator where there is one so no name is cut in half. A path of one long name is cut inside the name.
[[nodiscard]] std::string display_path(const std::filesystem::path& path, std::size_t max_chars);

}
