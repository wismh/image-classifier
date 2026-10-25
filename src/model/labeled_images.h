#pragma once

#include "model/pet_image.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace game {

// Normalized photos with a class each: `images` holds size() * kImageBytes bytes. `class_names` says what the two
// classes are.
struct LabeledImages {
    ClassNames class_names{"class 1", "class 2"};
    std::vector<std::uint8_t> images;
    std::vector<std::uint8_t> labels;

    [[nodiscard]] std::size_t size() const { return labels.size(); }
    [[nodiscard]] std::size_t count(ClassId class_id) const {
        return static_cast<std::size_t>(std::ranges::count(labels, static_cast<std::uint8_t>(class_id)));
    }
    [[nodiscard]] std::span<const std::uint8_t> image(std::size_t index) const {
        return std::span<const std::uint8_t>(images).subspan(index * kImageBytes, kImageBytes);
    }

    void add(std::span<const std::uint8_t> pixels, ClassId class_id) {
        images.insert(images.end(), pixels.begin(), pixels.end());
        labels.push_back(static_cast<std::uint8_t>(class_id));
    }
};

}
