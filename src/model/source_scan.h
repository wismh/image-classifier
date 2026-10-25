#pragma once

#include "model/pet_image.h"

#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace game {

struct SourceFile {
    std::filesystem::path path;
    ClassId class_id = ClassId::First;
};

struct SourceScan {
    std::vector<SourceFile> files;  // in path order
    ClassNames class_names;         // the names of the two subfolders, in the order of the names
};

// The photos of a source folder. The folder holds one subfolder for each of the two classes, named for it ("cats" and
// "dogs",
// "male" and "female"); the subfolders are the classes in the alphabetical order of their names, whatever the case.
// Photos
// (.jpg, .jpeg, .png, .bmp, .gif) are taken from them at any depth; files beside the subfolders and subfolders without
// photos are ignored. Fails when the folder is missing or does not have exactly two subfolders with photos.
[[nodiscard]] std::expected<SourceScan, std::string> scan_source_folder(const std::filesystem::path& folder);

}
