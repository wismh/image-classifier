#pragma once

#include "model/labeled_images.h"

#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace game {

// The datasets the app knows, one `<name>.cvds` file each in one directory. An empty directory path means there is no
// storage.
class DatasetStore {
public:
    static constexpr std::size_t kMaxNameLength = 32;

    explicit DatasetStore(std::filesystem::path directory);

    [[nodiscard]] bool available() const { return !directory_.empty(); }
    [[nodiscard]] bool contains(const std::string& name) const;
    // Names without the extension, sorted.
    [[nodiscard]] std::vector<std::string> list() const;
    [[nodiscard]] std::expected<LabeledImages, std::string> load(const std::string& name) const;
    // Writes the whole file: creates the dataset when it does not exist yet.
    [[nodiscard]] std::expected<void, std::string> save(const std::string& name, const LabeledImages& data) const;
    // Checks the normalized dataset file at `source` and stores a copy of it under `name`. Returns the number of
    // samples.
    [[nodiscard]] std::expected<std::size_t, std::string> add_file(const std::filesystem::path& source,
            const std::string& name) const;
    bool remove(const std::string& name) const;

    enum class NameProblem { None, Empty, TooLong, BadCharacters };
    // A name is letters, digits, '-' and '_' only, 1..kMaxNameLength long.
    [[nodiscard]] static NameProblem check_name(const std::string& name);
    // The file name of `path` without the extension, with every character a name cannot hold replaced by '_' and cut
    // to kMaxNameLength.
    [[nodiscard]] static std::string suggest_name(const std::filesystem::path& path);

private:
    [[nodiscard]] std::filesystem::path path_for(const std::string& name) const;

    std::filesystem::path directory_;
};

}
