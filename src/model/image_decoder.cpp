#include "model/image_decoder.h"

#include <format>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#include "../../third_party/stb/stb_image.h"

namespace game {

std::expected<RgbImage, std::string> decode_image(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        return std::unexpected(std::format("cannot open {}", path.string()));
    }
    const std::streamsize size = in.tellg();
    if (size <= 0) {
        return std::unexpected(std::format("{} is empty", path.string()));
    }
    std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(bytes.data()), size)) {
        return std::unexpected(std::format("cannot read {}", path.string()));
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* data = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()), &width, &height,
            &channels, 3);
    if (data == nullptr) {
        return std::unexpected(std::format("{} is not a readable image", path.string()));
    }
    RgbImage image;
    image.width = width;
    image.height = height;
    image.pixels.assign(data, data + static_cast<std::size_t>(width) * height * 3);
    stbi_image_free(data);
    return image;
}

}
