#include "model/model_file.h"

#include "model/binary_io.h"

#include <algorithm>
#include <format>
#include <fstream>

namespace game {

namespace {

constexpr char kMagic[4] = {'C', 'V', 'M', 'D'};
constexpr std::uint32_t kVersion = 3;
constexpr std::uint32_t kMaxNameBytes = 256;
// Version 1: magic, version, 4 filter/hidden sizes, epochs, accuracy, train samples, parameter count.
constexpr std::uintmax_t kHeaderBytesV1 = sizeof(kMagic) + 9 * sizeof(std::uint32_t);
// Version 2 adds the split seed and the dataset size before the parameter count.
constexpr std::uintmax_t kHeaderBytesV2 = kHeaderBytesV1 + 2 * sizeof(std::uint32_t);

}

std::expected<void, std::string> write_model_file(const std::filesystem::path& path, const ModelFile& model) {
    std::error_code error;
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), error);
    }
    const std::filesystem::path temporary = path.string() + ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary);
        if (!out) {
            return std::unexpected(std::format("cannot write {}", path.string()));
        }
        out.write(kMagic, sizeof(kMagic));
        write_value(out, kVersion);
        write_value(out, static_cast<std::uint32_t>(model.config.conv1));
        write_value(out, static_cast<std::uint32_t>(model.config.conv2));
        write_value(out, static_cast<std::uint32_t>(model.config.conv3));
        write_value(out, static_cast<std::uint32_t>(model.config.hidden));
        write_value(out, static_cast<std::uint32_t>(model.epochs));
        write_value(out, model.balanced_accuracy);
        write_value(out, static_cast<std::uint32_t>(model.train_samples));
        write_value(out, model.split_seed);
        write_value(out, static_cast<std::uint32_t>(model.dataset_samples));
        for (const std::string& name : model.class_names) {
            write_text(out, name);
        }
        write_value(out, static_cast<std::uint32_t>(model.params.size()));
        out.write(reinterpret_cast<const char*>(model.params.data()),
                static_cast<std::streamsize>(model.params.size() * sizeof(float)));
        if (!out) {
            return std::unexpected(std::format("write failed for {}", path.string()));
        }
    }
    std::filesystem::rename(temporary, path, error);
    if (error) {
        return std::unexpected(std::format("cannot replace {}: {}", path.string(), error.message()));
    }
    return {};
}

std::expected<ModelFile, std::string> read_model_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::unexpected(std::format("cannot open {}", path.string()));
    }
    char magic[4]{};
    std::uint32_t version = 0;
    std::uint32_t sizes[4]{};
    std::uint32_t epochs = 0;
    float accuracy = 0.0f;
    std::uint32_t train_samples = 0;
    std::uint32_t split_seed = 0;
    std::uint32_t dataset_samples = 0;
    std::uint32_t count = 0;
    in.read(magic, sizeof(magic));
    bool ok = static_cast<bool>(in) && read_value(in, version);
    for (std::uint32_t& size : sizes) {
        ok = ok && read_value(in, size);
    }
    ok = ok && read_value(in, epochs) && read_value(in, accuracy) && read_value(in, train_samples);
    if (ok && version >= 2) {
        ok = read_value(in, split_seed) && read_value(in, dataset_samples);
    }
    ClassNames names = legacy_class_names();
    std::uintmax_t names_bytes = 0;
    if (ok && version >= 3) {
        for (std::string& name : names) {
            ok = ok && read_text(in, name, kMaxNameBytes);
            names_bytes += text_bytes(name);
        }
    }
    ok = ok && read_value(in, count);
    if (!ok || !std::equal(std::begin(magic), std::end(magic), std::begin(kMagic)) || version < 1 || version > 3) {
        return std::unexpected(std::format("{} is not a model file", path.string()));
    }
    const std::uintmax_t header_bytes = (version == 1 ? kHeaderBytesV1 : kHeaderBytesV2) + names_bytes;

    // The header must agree with the file, or a damaged count would ask for gigabytes.
    std::error_code size_error;
    const std::uintmax_t file_size = std::filesystem::file_size(path, size_error);
    if (size_error || file_size != header_bytes + std::uintmax_t{count} * sizeof(float)) {
        return std::unexpected(std::format("{} is damaged", path.string()));
    }

    ModelFile model;
    model.config = CnnConfig{static_cast<int>(sizes[0]), static_cast<int>(sizes[1]), static_cast<int>(sizes[2]),
            static_cast<int>(sizes[3])};
    model.epochs = epochs;
    model.balanced_accuracy = accuracy;
    model.train_samples = train_samples;
    model.split_known = version >= 2;
    model.class_names = std::move(names);
    model.split_seed = split_seed;
    model.dataset_samples = dataset_samples;
    model.params.resize(count);
    in.read(reinterpret_cast<char*>(model.params.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!in) {
        return std::unexpected(std::format("{} is damaged", path.string()));
    }
    if (!model.config.valid() || !Cnn::with_params(model.config, model.params)) {
        return std::unexpected(std::format("{} holds a network this version cannot run", path.string()));
    }
    return model;
}

}
