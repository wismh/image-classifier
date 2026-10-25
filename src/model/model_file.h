#pragma once

#include "model/cnn.h"

#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace game {

// A trained network on disk, one `.cvm` file: a magic, a version, the size of the network, what it was trained on, then
// all of its parameters. Version 2 also records the seed of the split of the dataset and its size, so that training can
// go on later with the same validation part, and version 3 the names of the two classes. Older files are read too,
// without
// what they lack (their classes are cat and dog).
inline constexpr const char* kModelExtension = ".cvm";

struct ModelFile {
    CnnConfig config;
    unsigned epochs = 0;                // epochs the network was trained for
    float balanced_accuracy = 0.0f;     // on the validation part of the data it was trained on
    unsigned train_samples = 0;
    // Which samples were held out for checking: split_dataset(data, fraction, split_seed). Unknown (split_known false)
    // for a file written before version 2; `dataset_samples` is then 0 too.
    bool split_known = false;
    std::uint32_t split_seed = 0;
    unsigned dataset_samples = 0;       // the size of the whole dataset the split was made of
    ClassNames class_names = legacy_class_names();
    std::vector<float> params;
};

// Writes beside the final name and renames, so a failed write never leaves half a model.
[[nodiscard]] std::expected<void, std::string> write_model_file(const std::filesystem::path& path,
        const ModelFile& model);
[[nodiscard]] std::expected<ModelFile, std::string> read_model_file(const std::filesystem::path& path);

}
