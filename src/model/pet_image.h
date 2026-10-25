#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace game {

// Every normalized photo is a kImageSide x kImageSide RGB picture, rows top to bottom, three bytes per pixel.
constexpr int kImageSide = 64;
constexpr int kChannels = 3;
constexpr std::size_t kImageBytes = static_cast<std::size_t>(kImageSide) * kImageSide * kChannels;

// The network tells one thing from another: every picture belongs to one of two classes. Which two things they are
// ("cats" and "dogs", "male" and "female") is the data's business and goes by names that travel with the datasets and
// the networks.
enum class ClassId : std::uint8_t { First = 0, Second = 1 };
constexpr std::size_t kNumClasses = 2;

using ClassNames = std::array<std::string, kNumClasses>;

// The names of a dataset or a network that was written before names were recorded: those were all about cats and dogs.
inline ClassNames legacy_class_names() {
    return {"cat", "dog"};
}

}
