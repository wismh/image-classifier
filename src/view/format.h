#pragma once

#include <format>
#include <string>

namespace game {

// A share 0..1 as a whole percent, without the sign: "87".
[[nodiscard]] inline std::string percent_text(float share) {
    return std::format("{:.0f}", share * 100.0f);
}

}
