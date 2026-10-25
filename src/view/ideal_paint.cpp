#include "view/ideal_paint.h"

#include <engine/builtin_ids.h>

#include <algorithm>

namespace game {

void IdealPaint::paint(engine::ui::IDrawList& list, const engine::render::Rect& content) {
    const float side = std::min((content.w - kGap) * 0.5f, content.h - kCaptionHeight);
    const float pixel = side / kImageSide;
    for (std::size_t species = 0; species < kNumClasses; ++species) {
        const float left = content.x + static_cast<float>(species) * (side + kGap);
        list.fill_rect({left, content.y, side, side}, {0.12f, 0.16f, 0.23f, 1.0f});
        const std::vector<std::uint8_t>& picture = pictures_[species];
        if (!picture.empty()) {
            for (int y = 0; y < kImageSide; ++y) {
                for (int x = 0; x < kImageSide; ++x) {
                    const auto offset = static_cast<std::size_t>(y) * kImageSide + x;
                    const std::uint8_t* rgb = picture.data() + offset * kChannels;
                    list.fill_rect({left + x * pixel, content.y + y * pixel, pixel + 0.5f, pixel + 0.5f},
                            {rgb[0] / 255.0f, rgb[1] / 255.0f, rgb[2] / 255.0f, 1.0f});
                }
            }
        }
        list.set_font(engine::builtin::font_ui, 14.0f);
        list.text(names_[species], {left, content.y + side + 4.0f}, {0.2f, 0.25f, 0.33f, 1.0f});
    }
}

void IdealPaint::set(ClassId species, std::span<const std::uint8_t> picture) {
    pictures_[static_cast<std::size_t>(species)].assign(picture.begin(), picture.end());
    ++version_;
}

void IdealPaint::reset() {
    for (auto& picture : pictures_) {
        picture.clear();
    }
    ++version_;
}

void IdealPaint::set_names(std::string first, std::string second) {
    names_ = {std::move(first), std::move(second)};
    ++version_;
}

}
