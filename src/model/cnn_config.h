#pragma once

#include <string>

namespace game {

// The size of the network: three blocks of convolution + ReLU + 2x2 max pooling (64x64 -> 32x32 -> 16x16 -> 8x8), then
// a hidden dense layer with ReLU and a dense layer of kNumClasses scores.
struct CnnConfig {
    static constexpr int kMaxFilters = 64;
    static constexpr int kMaxHidden = 512;

    int conv1 = 8;
    int conv2 = 16;
    int conv3 = 32;
    int hidden = 64;

    [[nodiscard]] bool valid() const;
    // "8-16-32-64": the file name and the list say which network it is.
    [[nodiscard]] std::string label() const;
    bool operator==(const CnnConfig&) const = default;
};

}
