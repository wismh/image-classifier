#include "model/model_store.h"

#include <algorithm>
#include <chrono>
#include <format>

namespace game {

ModelStore::ModelStore(std::filesystem::path directory)
    : directory_(std::move(directory)) {}

std::filesystem::path ModelStore::path_for(const std::string& name) const {
    return directory_ / (name + kModelExtension);
}

std::vector<std::string> ModelStore::list() const {
    std::vector<std::string> names;
    if (!available()) {
        return names;
    }
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(directory_, error)) {
        if (entry.is_regular_file(error) && entry.path().extension() == kModelExtension) {
            names.push_back(entry.path().stem().string());
        }
    }
    std::ranges::sort(names);
    return names;
}

std::expected<ModelFile, std::string> ModelStore::load(const std::string& name) const {
    return read_model_file(path_for(name));
}

std::expected<void, std::string> ModelStore::save(const std::string& name, const ModelFile& model) const {
    if (!available()) {
        return std::unexpected("no user data directory");
    }
    return write_model_file(path_for(name), model);
}

bool ModelStore::remove(const std::string& name) const {
    std::error_code error;
    return available() && std::filesystem::remove(path_for(name), error);
}

std::string ModelStore::new_name(const CnnConfig& config, unsigned epochs) {
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch());
    return std::format("{}_{}ep_{}", config.label(), epochs, seconds.count());
}

}
