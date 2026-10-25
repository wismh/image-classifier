#include "model/augment.h"

#include "model/pet_image.h"

#include <algorithm>

namespace game {

namespace {

// splitmix64: mixes a few numbers into one well spread seed.
std::uint64_t mix(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

constexpr std::size_t kPlane = static_cast<std::size_t>(kImageSide) * kImageSide;

}

void to_input(std::span<const std::uint8_t> picture, std::span<float> input) {
    for (std::size_t i = 0; i < kPlane; ++i) {
        for (int c = 0; c < kChannels; ++c) {
            input[static_cast<std::size_t>(c) * kPlane + i] = static_cast<float>(picture[i * kChannels + c]) / 255.0f - 0.5f;
        }
    }
}

void to_augmented_input(std::span<const std::uint8_t> picture, std::span<float> input,
        const Augmentation& augmentation, std::mt19937& rng) {
    const bool flip = augmentation.flip && (rng() & 1u) != 0;
    const auto draw_int = [&rng](int low, int high) {
        return low + static_cast<int>(rng() % static_cast<unsigned>(high - low + 1));
    };
    const auto draw_float = [&rng](float amount) {
        return amount * (static_cast<float>(rng() & 0xFFFFu) / 32767.5f - 1.0f);
    };
    const int shift_x = augmentation.max_shift > 0 ? draw_int(-augmentation.max_shift, augmentation.max_shift) : 0;
    const int shift_y = augmentation.max_shift > 0 ? draw_int(-augmentation.max_shift, augmentation.max_shift) : 0;
    const float brightness = draw_float(augmentation.brightness);
    const float contrast = 1.0f + draw_float(augmentation.contrast);

    for (int y = 0; y < kImageSide; ++y) {
        const int source_y = std::clamp(y + shift_y, 0, kImageSide - 1);
        for (int x = 0; x < kImageSide; ++x) {
            const int moved_x = std::clamp(x + shift_x, 0, kImageSide - 1);
            const int source_x = flip ? kImageSide - 1 - moved_x : moved_x;
            const std::size_t from = (static_cast<std::size_t>(source_y) * kImageSide + source_x) * kChannels;
            for (int c = 0; c < kChannels; ++c) {
                const float value = static_cast<float>(picture[from + c]) / 255.0f - 0.5f;
                input[static_cast<std::size_t>(c) * kPlane + static_cast<std::size_t>(y) * kImageSide + x] =
                        std::clamp(value * contrast + brightness, -0.5f, 0.5f);
            }
        }
    }
}

std::mt19937 sample_rng(std::uint32_t seed, unsigned epoch, std::size_t sample) {
    const std::uint64_t mixed = mix(mix(mix(seed) + epoch) + sample);
    return std::mt19937{static_cast<std::uint32_t>(mixed ^ (mixed >> 32))};
}

}
