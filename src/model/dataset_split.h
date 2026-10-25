#pragma once

#include "model/labeled_images.h"

#include <cstdint>
#include <vector>

namespace game {

// Which samples of a dataset train the model and which only check it. Both lists hold indices into the dataset, in
// ascending order, and together they hold every index once.
struct DatasetSplit {
    std::vector<std::size_t> train;
    std::vector<std::size_t> validation;
};

// A random split that keeps the share of each class the same in both parts: each species gives
// round(count * validation_fraction) samples to validation. A species with two or more samples always keeps at least
// one in each part, so neither part loses a class. The same dataset, fraction and seed always give the same split, on
// every platform. `validation_fraction` is clamped to [0, 1].
[[nodiscard]] DatasetSplit split_dataset(const LabeledImages& data, double validation_fraction, std::uint32_t seed);

}
