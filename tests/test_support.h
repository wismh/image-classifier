#pragma once

#include "model/labeled_images.h"
#include "model/parallel.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <random>
#include <thread>
#include <string>
#include <vector>

namespace game::test {

// A directory that exists for the lifetime of the object.
class TempDir {
public:
    TempDir() {
        static std::uint32_t counter = 0;
        path_ = std::filesystem::temp_directory_path()
                / ("pets_tests_" + std::to_string(std::random_device{}()) + "_" + std::to_string(counter++));
        std::filesystem::create_directories(path_);
    }
    ~TempDir() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const { return path_; }
    [[nodiscard]] std::filesystem::path operator/(const std::string& name) const { return path_ / name; }

private:
    std::filesystem::path path_;
};

using Rgb = std::array<std::uint8_t, 3>;

// Writes a 24-bit BMP of width x height whose pixel at (x, y) is color(x, y).
template<typename Color>
void write_bmp(const std::filesystem::path& path, int width, int height, Color color) {
    std::filesystem::create_directories(path.parent_path());
    const int row_bytes = (width * 3 + 3) / 4 * 4;
    const std::uint32_t pixel_bytes = static_cast<std::uint32_t>(row_bytes * height);
    std::vector<std::uint8_t> bytes;
    const auto put = [&bytes](auto value) {
        const auto* raw = reinterpret_cast<const std::uint8_t*>(&value);
        bytes.insert(bytes.end(), raw, raw + sizeof(value));
    };
    bytes.push_back('B');
    bytes.push_back('M');
    put(std::uint32_t{54 + pixel_bytes});
    put(std::uint32_t{0});
    put(std::uint32_t{54});
    put(std::uint32_t{40});
    put(std::int32_t{width});
    put(std::int32_t{height});
    put(std::uint16_t{1});
    put(std::uint16_t{24});
    put(std::uint32_t{0});
    put(pixel_bytes);
    put(std::int32_t{2835});
    put(std::int32_t{2835});
    put(std::uint32_t{0});
    put(std::uint32_t{0});
    for (int y = height - 1; y >= 0; --y) {  // BMP rows run bottom to top
        const std::size_t start = bytes.size();
        for (int x = 0; x < width; ++x) {
            const Rgb rgb = color(x, y);
            bytes.push_back(rgb[2]);
            bytes.push_back(rgb[1]);
            bytes.push_back(rgb[0]);
        }
        bytes.resize(start + static_cast<std::size_t>(row_bytes), 0);
    }
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

inline void write_bmp(const std::filesystem::path& path, int width, int height, Rgb rgb) {
    write_bmp(path, width, height, [rgb](int, int) { return rgb; });
}

inline void write_text(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out << text;
}

// A few pictures with a pattern the label can be recognized by: a cat is dark, a dog is bright.
inline LabeledImages synthetic_set(std::size_t count, std::uint32_t seed) {
    std::mt19937 rng{seed};
    std::uniform_int_distribution<int> noise(0, 30);
    LabeledImages data;
    for (std::size_t i = 0; i < count; ++i) {
        const ClassId species = i % 2 == 0 ? ClassId::First : ClassId::Second;
        std::vector<std::uint8_t> pixels(kImageBytes);
        for (auto& pixel : pixels) {
            pixel = static_cast<std::uint8_t>((species == ClassId::First ? 40 : 200) + noise(rng));
        }
        data.add(pixels, species);
    }
    return data;
}

inline void write_bytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

template<typename T>
void append(std::vector<std::uint8_t>& bytes, const T& value) {
    const auto* raw = reinterpret_cast<const std::uint8_t*>(&value);
    bytes.insert(bytes.end(), raw, raw + sizeof(T));
}

// Four threads, as the engine's job system would run it.
inline Parallel threaded() {
    return Parallel{4, [](std::size_t count, std::size_t grain, const Parallel::RangeFn& fn) {
        std::vector<std::jthread> threads;
        for (std::size_t begin = 0; begin < count; begin += grain) {
            threads.emplace_back([&fn, begin, end = std::min(count, begin + grain)] { fn(begin, end); });
        }
    }};
}

}
