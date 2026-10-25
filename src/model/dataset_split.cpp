#include "model/dataset_split.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>

namespace game {

namespace {

// Fisher-Yates over mt19937, whose output the standard fixes: std::shuffle may order differently per standard library.
void shuffle(std::vector<std::size_t>& items, std::mt19937& rng) {
    for (std::size_t i = items.size(); i > 1; --i) {
        std::swap(items[i - 1], items[rng() % i]);
    }
}

std::size_t validation_count(std::size_t count, double fraction) {
    const auto wanted = static_cast<std::size_t>(std::llround(static_cast<double>(count) * fraction));
    if (count < 2) {
        return std::min(wanted, count);
    }
    return std::clamp<std::size_t>(wanted, fraction > 0.0 ? 1 : 0, fraction < 1.0 ? count - 1 : count);
}

}

DatasetSplit split_dataset(const LabeledImages& data, double validation_fraction, std::uint32_t seed) {
    const double fraction = std::clamp(validation_fraction, 0.0, 1.0);

    std::array<std::vector<std::size_t>, kNumClasses> by_class;
    for (std::size_t index = 0; index < data.size(); ++index) {
        by_class[data.labels[index]].push_back(index);
    }

    std::mt19937 rng{seed};
    DatasetSplit split;
    for (auto& indices : by_class) {
        shuffle(indices, rng);
        const std::size_t held_out = validation_count(indices.size(), fraction);
        split.validation.insert(split.validation.end(), indices.begin(), indices.begin() + held_out);
        split.train.insert(split.train.end(), indices.begin() + held_out, indices.end());
    }
    std::ranges::sort(split.train);
    std::ranges::sort(split.validation);
    return split;
}

}
