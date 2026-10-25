#pragma once

#include "model/model_file.h"

#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace game {

// The saved models of the app: `<name>.cvm` files in one directory. An empty directory path means there is no storage.
class ModelStore {
public:
    explicit ModelStore(std::filesystem::path directory);

    [[nodiscard]] bool available() const { return !directory_.empty(); }
    // Names without the extension, sorted.
    [[nodiscard]] std::vector<std::string> list() const;
    [[nodiscard]] std::expected<ModelFile, std::string> load(const std::string& name) const;
    [[nodiscard]] std::expected<void, std::string> save(const std::string& name, const ModelFile& model) const;
    bool remove(const std::string& name) const;

    // `<network label>_<epochs>ep_<unix time>`: a name no other save of the app has.
    [[nodiscard]] static std::string new_name(const CnnConfig& config, unsigned epochs);

private:
    [[nodiscard]] std::filesystem::path path_for(const std::string& name) const;

    std::filesystem::path directory_;
};

}
