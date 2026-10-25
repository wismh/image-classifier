#include "view/sample_paint.h"

namespace game {

void SamplePaint::paint(engine::ui::IDrawList& list, const engine::render::Rect& content) {
    list.fill_rect(content, {0.12f, 0.16f, 0.23f, 1.0f});
    if (picture_.empty()) {
        return;
    }
    const float pixel = content.w / kImageSide;
    for (int y = 0; y < kImageSide; ++y) {
        for (int x = 0; x < kImageSide; ++x) {
            const std::uint8_t* rgb = picture_.data() + (static_cast<std::size_t>(y) * kImageSide + x) * kChannels;
            list.fill_rect({content.x + x * pixel, content.y + y * pixel, pixel + 0.5f, pixel + 0.5f},
                    {rgb[0] / 255.0f, rgb[1] / 255.0f, rgb[2] / 255.0f, 1.0f});
        }
    }
}

void SamplePaint::show(std::span<const std::uint8_t> picture) {
    picture_.assign(picture.begin(), picture.end());
    ++version_;
}

void SamplePaint::clear() {
    picture_.clear();
    ++version_;
}

}
