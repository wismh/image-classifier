#pragma once

#include "model/labeled_images.h"
#include "model/model_file.h"

#include <cstdint>

namespace game {

// How a saved network meets the dataset it is to be trained further on.
enum class ContinueKind {
    SameData,      // the dataset it was trained on, with a known split
    UnknownSplit,  // a file from before the split was recorded: the check samples cannot be kept apart
    OtherDataset,  // a dataset of another size: the recorded split no longer matches it
    OtherClasses,  // the two classes are not the ones the network was trained to tell apart: it cannot go on
};

struct Continuation {
    ContinueKind kind = ContinueKind::SameData;
    // The split to train with: the recorded one keeps the check samples out of the training, otherwise `fresh_seed`.
    std::uint32_t split_seed = 0;
};

[[nodiscard]] Continuation plan_continuation(const ModelFile& file, const LabeledImages& data,
        std::uint32_t fresh_seed);

}
