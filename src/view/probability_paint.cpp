#include "view/probability_paint.h"

#include <engine/builtin_ids.h>

#include <algorithm>
#include <cmath>
#include <format>

namespace game {

namespace {

constexpr float kNameWidth = 56.0f;
constexpr float kPercentWidth = 48.0f;
constexpr float kBarHeight = 18.0f;

void bar(engine::ui::IDrawList& list, const engine::render::Rect& row, const std::string& name, bool present,
        int percent, const glm::vec4& color) {
    const float track_x = row.x + kNameWidth;
    const float track_w = row.w - kNameWidth - kPercentWidth;
    const float top = row.y + (row.h - kBarHeight) * 0.5f;
    list.set_font(engine::builtin::font_ui, 14.0f);
    list.text(name, {row.x, top + 1.0f}, {0.2f, 0.25f, 0.33f, 1.0f});
    list.fill_rect({track_x, top, track_w, kBarHeight}, {0.89f, 0.91f, 0.94f, 1.0f});
    if (present) {
        list.fill_rect({track_x, top, track_w * static_cast<float>(percent) / 100.0f, kBarHeight}, color);
        list.text(std::format("{}%", percent), {track_x + track_w + 8.0f, top + 1.0f}, {0.06f, 0.09f, 0.16f, 1.0f});
    }
}

}

void ProbabilityPaint::paint(engine::ui::IDrawList& list, const engine::render::Rect& content) {
    const float half = content.h * 0.5f;
    bar(list, {content.x, content.y, content.w, half}, first_name_, present_, first_percent_,
            {0.98f, 0.62f, 0.15f, 1.0f});
    bar(list, {content.x, content.y + half, content.w, half}, second_name_, present_, second_percent_,
            {0.25f, 0.55f, 0.98f, 1.0f});
}

void ProbabilityPaint::show(float first, float second) {
    const int cat_percent = static_cast<int>(std::lround(std::clamp(first, 0.0f, 1.0f) * 100.0f));
    const int dog_percent = static_cast<int>(std::lround(std::clamp(second, 0.0f, 1.0f) * 100.0f));
    if (!present_ || cat_percent != first_percent_ || dog_percent != second_percent_) {
        present_ = true;
        first_percent_ = cat_percent;
        second_percent_ = dog_percent;
        ++version_;
    }
}

void ProbabilityPaint::clear() {
    if (present_) {
        present_ = false;
        ++version_;
    }
}

void ProbabilityPaint::set_names(std::string first, std::string second) {
    first_name_ = std::move(first);
    second_name_ = std::move(second);
    ++version_;
}

}
