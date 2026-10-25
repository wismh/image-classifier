#include "model/dataset_file.h"

#include "model/binary_io.h"

#include <algorithm>
#include <format>
#include <fstream>

namespace game {

namespace {

constexpr char kMagic[4] = {'C', 'V', 'D', 'S'};
// Version 1 has no names (its classes are cat and dog); version 2 puts the two class names after the number of samples.
constexpr std::uint32_t kVersion = 2;
constexpr std::uint32_t kMaxNameBytes = 256;
constexpr std::uintmax_t kHeaderBytes = sizeof(kMagic) + 4 * sizeof(std::uint32_t);

}

std::expected<void, std::string> write_dataset_file(const std::filesystem::path& path, const LabeledImages& data) {
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
        write_value(out, static_cast<std::uint32_t>(kImageSide));
        write_value(out, static_cast<std::uint32_t>(kChannels));
        write_value(out, static_cast<std::uint32_t>(data.size()));
        for (const std::string& name : data.class_names) {
            write_text(out, name);
        }
        out.write(reinterpret_cast<const char*>(data.labels.data()), static_cast<std::streamsize>(data.labels.size()));
        out.write(reinterpret_cast<const char*>(data.images.data()), static_cast<std::streamsize>(data.images.size()));
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

std::expected<LabeledImages, std::string> read_dataset_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::unexpected(std::format("cannot open {}", path.string()));
    }
    char magic[4]{};
    std::uint32_t version = 0;
    std::uint32_t side = 0;
    std::uint32_t channels = 0;
    std::uint32_t count = 0;
    in.read(magic, sizeof(magic));
    if (!in || !read_value(in, version) || !read_value(in, side) || !read_value(in, channels) || !read_value(in, count)
            || !std::equal(std::begin(magic), std::end(magic), std::begin(kMagic)) || (version != 1 && version != 2)) {
        return std::unexpected(std::format("{} is not a normalized dataset", path.string()));
    }
    if (side != kImageSide || channels != kChannels) {
        return std::unexpected(std::format("{} holds {}x{}x{} pictures, expected {}x{}x{}", path.string(), side, side,
                channels, kImageSide, kImageSide, kChannels));
    }

    LabeledImages data;
    data.class_names = legacy_class_names();
    std::uintmax_t header_bytes = kHeaderBytes;
    if (version == 2) {
        for (std::string& name : data.class_names) {
            if (!read_text(in, name, kMaxNameBytes)) {
                return std::unexpected(std::format("{} is damaged", path.string()));
            }
            header_bytes += text_bytes(name);
        }
    }

    // The header must agree with the file, or a damaged count would ask for gigabytes.
    std::error_code size_error;
    const std::uintmax_t file_size = std::filesystem::file_size(path, size_error);
    if (size_error || file_size != header_bytes + std::uintmax_t{count} * (1 + kImageBytes)) {
        return std::unexpected(std::format("{} is damaged", path.string()));
    }

    data.labels.resize(count);
    data.images.resize(static_cast<std::size_t>(count) * kImageBytes);
    in.read(reinterpret_cast<char*>(data.labels.data()), static_cast<std::streamsize>(data.labels.size()));
    in.read(reinterpret_cast<char*>(data.images.data()), static_cast<std::streamsize>(data.images.size()));
    if (!in || std::ranges::any_of(data.labels, [](std::uint8_t label) { return label >= kNumClasses; })) {
        return std::unexpected(std::format("{} is damaged", path.string()));
    }
    return data;
}

}
