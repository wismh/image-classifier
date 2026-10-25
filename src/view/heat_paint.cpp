#include "view/heat_paint.h"

#include <algorithm>
#include <cmath>

namespace game {

float HeatPaint::heat_at(int x, int y) const {
    // Cell centers are half a cell in; a pixel's coordinate in cells is shifted the same way.
    const float scale = static_cast<float>(side_) / kImageSide;
    const float fx = std::clamp((static_cast<float>(x) + 0.5f) * scale - 0.5f, 0.0f, static_cast<float>(side_ - 1));
    const float fy = std::clamp((static_cast<float>(y) + 0.5f) * scale - 0.5f, 0.0f, static_cast<float>(side_ - 1));
    const int x0 = static_cast<int>(fx);
    const int y0 = static_cast<int>(fy);
    const int x1 = std::min(x0 + 1, side_ - 1);
    const int y1 = std::min(y0 + 1, side_ - 1);
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const auto cell = [this](int cx, int cy) { return map_[static_cast<std::size_t>(cy) * side_ + cx]; };
    const float top = cell(x0, y0) * (1.0f - tx) + cell(x1, y0) * tx;
    const float bottom = cell(x0, y1) * (1.0f - tx) + cell(x1, y1) * tx;
    return top * (1.0f - ty) + bottom * ty;
}

void HeatPaint::paint(engine::ui::IDrawList& list, const engine::render::Rect& content) {
    list.fill_rect(content, {0.12f, 0.16f, 0.23f, 1.0f});
    if (picture_.empty()) {
        return;
    }
    const bool heated = side_ > 0;
    const float pixel = content.w / kImageSide;
    for (int y = 0; y < kImageSide; ++y) {
        for (int x = 0; x < kImageSide; ++x) {
            const std::uint8_t* rgb = picture_.data() + (static_cast<std::size_t>(y) * kImageSide + x) * kChannels;
            float r = rgb[0] / 255.0f;
            float g = rgb[1] / 255.0f;
            float b = rgb[2] / 255.0f;
            if (heated) {
                const float heat = std::clamp(heat_at(x, y), 0.0f, 1.0f);
                // Yellow where the evidence begins, red where it is strongest; the stronger, the less of the picture
                // shows.
                const float cover = 0.85f * std::pow(heat, 0.8f);
                const float dim = 0.75f;
                r = r * dim * (1.0f - cover) + 1.0f * cover;
                g = g * dim * (1.0f - cover) + (1.0f - heat) * cover;
                b = b * dim * (1.0f - cover);
            }
            list.fill_rect({content.x + x * pixel, content.y + y * pixel, pixel + 0.5f, pixel + 0.5f}, {r, g, b, 1.0f});
        }
    }
}

void HeatPaint::show(std::span<const std::uint8_t> picture, std::span<const float> map, int side) {
    picture_.assign(picture.begin(), picture.end());
    map_.assign(map.begin(), map.end());
    side_ = side;
    ++version_;
}

void HeatPaint::show_picture(std::span<const std::uint8_t> picture) {
    picture_.assign(picture.begin(), picture.end());
    map_.clear();
    side_ = 0;
    ++version_;
}

void HeatPaint::clear() {
    picture_.clear();
    map_.clear();
    side_ = 0;
    ++version_;
}

}
