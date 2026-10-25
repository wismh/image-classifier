#include "model/picture_file.h"

#include "model/image_decoder.h"
#include "model/image_resize.h"

namespace game {

std::expected<std::vector<std::uint8_t>, std::string> load_normalized_picture(const std::filesystem::path& path) {
    const auto image = decode_image(path);
    if (!image) {
        return std::unexpected(image.error());
    }
    return to_square_picture(*image);
}

}
