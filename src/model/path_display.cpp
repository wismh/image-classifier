#include "model/path_display.h"

#include <algorithm>

namespace game {

std::string display_path(const std::filesystem::path& path, std::size_t max_chars) {
    const std::u8string utf8 = path.u8string();
    const std::string text{utf8.begin(), utf8.end()};
    constexpr std::string_view kEllipsis = "...";
    if (text.size() <= max_chars || max_chars <= kEllipsis.size()) {
        return text;
    }
    const std::size_t keep = max_chars - kEllipsis.size();
    std::size_t start = text.size() - keep;
    // Start after the first separator that leaves the tail within the limit, if the tail holds one.
    const std::size_t separator = text.find_first_of("/\\", start);
    if (separator != std::string::npos && separator + 1 < text.size()) {
        start = separator;
    }
    // Never start inside a multi-byte UTF-8 character.
    while (start < text.size() && (static_cast<unsigned char>(text[start]) & 0xC0) == 0x80) {
        ++start;
    }
    return std::string{kEllipsis} + text.substr(start);
}

}
