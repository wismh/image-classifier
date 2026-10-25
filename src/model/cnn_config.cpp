#include "model/cnn_config.h"

#include <format>

namespace game {

bool CnnConfig::valid() const {
    const auto filters_ok = [](int n) { return n >= 1 && n <= kMaxFilters; };
    return filters_ok(conv1) && filters_ok(conv2) && filters_ok(conv3) && hidden >= 1 && hidden <= kMaxHidden;
}

std::string CnnConfig::label() const {
    return std::format("{}-{}-{}-{}", conv1, conv2, conv3, hidden);
}

}
