#include "model/source_scan.h"

#include <algorithm>
#include <cctype>
#include <format>

namespace game {

namespace {

std::string lower(std::string text) {
    std::ranges::transform(text, text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

bool is_photo(const std::filesystem::path& path) {
    const std::string extension = lower(path.extension().string());
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp"
            || extension == ".gif";
}

struct ClassFolder {
    std::string name;
    std::vector<std::filesystem::path> photos;
};

}

std::expected<SourceScan, std::string> scan_source_folder(const std::filesystem::path& folder) {
    std::error_code error;
    if (!std::filesystem::is_directory(folder, error)) {
        return std::unexpected(std::format("{} is not a folder", folder.string()));
    }

    std::vector<ClassFolder> folders;
    for (const auto& entry : std::filesystem::directory_iterator(folder, error)) {
        if (!entry.is_directory(error)) {
            continue;
        }
        ClassFolder found{entry.path().filename().string(), {}};
        for (const auto& item : std::filesystem::recursive_directory_iterator(entry.path(), error)) {
            if (item.is_regular_file(error) && is_photo(item.path())) {
                found.photos.push_back(item.path());
            }
        }
        if (!found.photos.empty()) {
            folders.push_back(std::move(found));
        }
    }

    if (folders.size() != kNumClasses) {
        return std::unexpected(std::format("the folder needs exactly two subfolders with photos in them, one for each of the "
                                           "two classes, and has {}", folders.size()));
    }
    std::ranges::sort(folders, [](const ClassFolder& a, const ClassFolder& b) {
        return lower(a.name) != lower(b.name) ? lower(a.name) < lower(b.name) : a.name < b.name;
    });

    SourceScan scan;
    for (std::size_t i = 0; i < folders.size(); ++i) {
        scan.class_names[i] = folders[i].name;
        for (auto& photo : folders[i].photos) {
            scan.files.push_back({std::move(photo), static_cast<ClassId>(i)});
        }
    }
    std::ranges::sort(scan.files, {}, &SourceFile::path);
    return scan;
}

}
