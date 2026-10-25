#pragma once

#include <bit>
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>

namespace game {

static_assert(std::endian::native == std::endian::little, "model and dataset files are little-endian");

// Reads the raw bytes of a trivially copyable value; false when the stream ran out.
template<typename T>
bool read_value(std::istream& in, T& value) {
    return static_cast<bool>(in.read(reinterpret_cast<char*>(&value), sizeof(T)));
}

// A short text: its length, then its bytes. Longer than `max_length` counts as damaged.
inline bool read_text(std::istream& in, std::string& text, std::uint32_t max_length = 256) {
    std::uint32_t length = 0;
    if (!read_value(in, length) || length > max_length) {
        return false;
    }
    text.resize(length);
    return length == 0 || static_cast<bool>(in.read(text.data(), static_cast<std::streamsize>(length)));
}

template<typename T>
void write_value(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

inline void write_text(std::ostream& out, const std::string& text) {
    write_value(out, static_cast<std::uint32_t>(text.size()));
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

// The bytes write_text() puts down for `text`.
inline std::uintmax_t text_bytes(const std::string& text) {
    return sizeof(std::uint32_t) + text.size();
}

}
