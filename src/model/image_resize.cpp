#include "model/image_resize.h"

#include "model/pet_image.h"

#include <algorithm>
#include <cmath>

namespace game {

namespace {

struct Taps {
    int first = 0;
    std::vector<float> weights;
};

// For each of `dst` outputs, the source pixels of [offset, offset + src) its box covers and how much of each: the
// exact overlap, so the weights of one output sum to 1.
std::vector<Taps> box_taps(int src, int offset, int dst) {
    const double scale = static_cast<double>(src) / dst;
    std::vector<Taps> taps(static_cast<std::size_t>(dst));
    for (int d = 0; d < dst; ++d) {
        const double low = d * scale;
        const double high = (d + 1) * scale;
        const int first = static_cast<int>(std::floor(low));
        const int last = std::min(src, static_cast<int>(std::ceil(high)));
        Taps& tap = taps[static_cast<std::size_t>(d)];
        tap.first = offset + first;
        for (int k = first; k < last; ++k) {
            const double overlap = std::min(high, k + 1.0) - std::max(low, static_cast<double>(k));
            tap.weights.push_back(static_cast<float>(overlap / scale));
        }
    }
    return taps;
}

}

std::vector<std::uint8_t> to_square_picture(const RgbImage& image) {
    std::vector<std::uint8_t> out(kImageBytes, 0);
    const int side = std::min(image.width, image.height);
    if (side <= 0) {
        return out;
    }
    const auto columns = box_taps(side, (image.width - side) / 2, kImageSide);
    const auto rows = box_taps(side, (image.height - side) / 2, kImageSide);
    const std::size_t stride = static_cast<std::size_t>(image.width) * kChannels;

    // Horizontal pass over only the source rows the vertical pass needs.
    const int top = rows.front().first;
    const int bottom = rows.back().first + static_cast<int>(rows.back().weights.size());
    std::vector<float> wide(static_cast<std::size_t>(bottom - top) * kImageSide * kChannels, 0.0f);
    for (int y = top; y < bottom; ++y) {
        const std::uint8_t* row = image.pixels.data() + static_cast<std::size_t>(y) * stride;
        float* target = wide.data() + static_cast<std::size_t>(y - top) * kImageSide * kChannels;
        for (int x = 0; x < kImageSide; ++x) {
            const Taps& tap = columns[static_cast<std::size_t>(x)];
            for (std::size_t k = 0; k < tap.weights.size(); ++k) {
                const std::uint8_t* pixel = row + (static_cast<std::size_t>(tap.first) + k) * kChannels;
                for (int c = 0; c < kChannels; ++c) {
                    target[x * kChannels + c] += tap.weights[k] * pixel[c];
                }
            }
        }
    }

    for (int y = 0; y < kImageSide; ++y) {
        const Taps& tap = rows[static_cast<std::size_t>(y)];
        for (int i = 0; i < kImageSide * kChannels; ++i) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < tap.weights.size(); ++k) {
                sum += tap.weights[k]
                        * wide[(static_cast<std::size_t>(tap.first - top) + k) * kImageSide * kChannels + i];
            }
            out[static_cast<std::size_t>(y) * kImageSide * kChannels + i]
                    = static_cast<std::uint8_t>(std::clamp(sum + 0.5f, 0.0f, 255.0f));
        }
    }
    return out;
}

}
