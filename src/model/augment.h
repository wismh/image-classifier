#pragma once

#include <cstdint>
#include <random>
#include <span>

namespace game {

// Random changes to a picture shown to the network during training, new ones every time a sample is shown.
struct Augmentation {
    bool flip = true;              // mirror left to right, half of the time
    int max_shift = 4;             // move by up to this many pixels each way (the edge is stretched over the gap)
    float brightness = 0.12f;      // add up to this much (1 = the whole range) to every value
    float contrast = 0.15f;        // stretch the values around the middle by up to this share

    [[nodiscard]] static Augmentation none() { return {false, 0, 0.0f, 0.0f}; }
};

// Turns a normalized picture (rows of RGB bytes, kImageSide x kImageSide) into the network's input (channel-major
// floats): a byte b becomes b / 255 - 0.5, so values lie in [-0.5, 0.5].
void to_input(std::span<const std::uint8_t> picture, std::span<float> input);

// The same, after a random flip, shift, brightness and contrast change drawn from `rng`.
void to_augmented_input(std::span<const std::uint8_t> picture, std::span<float> input,
        const Augmentation& augmentation, std::mt19937& rng);

// A generator for one showing of one sample: depends only on its arguments, so a run gives the same result however
// its work is spread over threads.
[[nodiscard]] std::mt19937 sample_rng(std::uint32_t seed, unsigned epoch, std::size_t sample);

}
