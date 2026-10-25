#include "model/output_path.h"

#include "model/dataset_file.h"

namespace game {

std::filesystem::path planned_output_path(const std::filesystem::path& source,
        const std::filesystem::path& output_directory) {
    if (source.empty()) {
        return {};
    }
    // "C:/photos/" has no file name of its own: its last folder is the name.
    const std::filesystem::path name = source.has_filename() ? source.filename() : source.parent_path().filename();
    const std::filesystem::path directory = output_directory.empty() ? source.parent_path() : output_directory;
    std::filesystem::path file = name.empty() ? std::filesystem::path{"photos"} : name;
    file += kDatasetExtension;
    return directory / file;
}

}
