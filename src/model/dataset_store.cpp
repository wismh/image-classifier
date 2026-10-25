#include "model/dataset_store.h"

#include "model/dataset_file.h"

#include <algorithm>
#include <cctype>

namespace game {

DatasetStore::DatasetStore(std::filesystem::path directory)
    : directory_(std::move(directory)) {}

std::filesystem::path DatasetStore::path_for(const std::string& name) const {
    return directory_ / (name + kDatasetExtension);
}

bool DatasetStore::contains(const std::string& name) const {
    std::error_code error;
    return available() && std::filesystem::is_regular_file(path_for(name), error);
}

std::vector<std::string> DatasetStore::list() const {
    std::vector<std::string> names;
    if (!available()) {
        return names;
    }
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(directory_, error)) {
        if (entry.is_regular_file(error) && entry.path().extension() == kDatasetExtension) {
            names.push_back(entry.path().stem().string());
        }
    }
    std::ranges::sort(names);
    return names;
}

std::expected<LabeledImages, std::string> DatasetStore::load(const std::string& name) const {
    return read_dataset_file(path_for(name));
}

std::expected<void, std::string> DatasetStore::save(const std::string& name, const LabeledImages& data) const {
    if (!available()) {
        return std::unexpected("no user data directory");
    }
    return write_dataset_file(path_for(name), data);
}

std::expected<std::size_t, std::string> DatasetStore::add_file(const std::filesystem::path& source,
        const std::string& name) const {
    auto data = read_dataset_file(source);
    if (!data) {
        return std::unexpected(std::move(data.error()));
    }
    if (auto saved = save(name, *data); !saved) {
        return std::unexpected(std::move(saved.error()));
    }
    return data->size();
}

bool DatasetStore::remove(const std::string& name) const {
    std::error_code error;
    return available() && std::filesystem::remove(path_for(name), error);
}

DatasetStore::NameProblem DatasetStore::check_name(const std::string& name) {
    if (name.empty()) {
        return NameProblem::Empty;
    }
    if (name.size() > kMaxNameLength) {
        return NameProblem::TooLong;
    }
    const bool plain = std::ranges::all_of(name, [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_';
    });
    return plain ? NameProblem::None : NameProblem::BadCharacters;
}

std::string DatasetStore::suggest_name(const std::filesystem::path& path) {
    std::string name = path.stem().string();
    for (char& c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') {
            c = '_';
        }
    }
    name.resize(std::min(name.size(), kMaxNameLength));
    return name;
}

}
