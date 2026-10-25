#include "view/filter_paint.h"

#include <algorithm>
#include <cmath>

namespace game {

namespace {

constexpr int kMeterSteps = 16;
constexpr float kMeterHeight = 3.0f;

// Weights are mostly small and a few are large; the root keeps the small ones visible. Positive is red, negative blue.
glm::vec4 signed_color(float value) {
    const float t = std::sqrt(std::min(1.0f, std::abs(value)));
    return value > 0.0f ? glm::vec4{t, 0.22f * t, 0.12f * t, 1.0f} : glm::vec4{0.12f * t, 0.45f * t, t, 1.0f};
}

}

void FilterPaint::paint(engine::ui::IDrawList& list, const engine::render::Rect& content) {
    if (vacant_) {
        return;
    }
    list.fill_rect(content, image_.kind == CellImage::Kind::None ? glm::vec4{0.12f, 0.16f, 0.23f, 1.0f}
                                                                 : glm::vec4{0.0f, 0.0f, 0.0f, 1.0f});
    if (image_.kind != CellImage::Kind::None) {
        const float pixel = std::min(content.w / static_cast<float>(image_.width),
                content.h / static_cast<float>(image_.height));
        const float left = content.x + (content.w - pixel * static_cast<float>(image_.width)) * 0.5f;
        const float top = content.y + (content.h - pixel * static_cast<float>(image_.height)) * 0.5f;
        const int channels = image_.kind == CellImage::Kind::Color ? 3 : 1;
        for (int y = 0; y < image_.height; ++y) {
            for (int x = 0; x < image_.width; ++x) {
                const float* value = image_.values.data() + (static_cast<std::size_t>(y) * image_.width + x) * channels;
                glm::vec4 color;
                switch (image_.kind) {
                case CellImage::Kind::Gray: color = {value[0], value[0], value[0], 1.0f}; break;
                case CellImage::Kind::Signed: color = signed_color(value[0]); break;
                default: color = {value[0], value[1], value[2], 1.0f}; break;
                }
                // A little over the size of a pixel, so that rounding leaves no seams between neighbours.
                list.fill_rect({left + static_cast<float>(x) * pixel, top + static_cast<float>(y) * pixel, pixel + 0.5f,
                                       pixel + 0.5f},
                        color);
            }
        }
    }
    if (meter_steps_ > 0) {
        const float width = content.w * static_cast<float>(meter_steps_) / kMeterSteps;
        const glm::vec4 color = meter_kind_ == MeterKind::First   ? glm::vec4{0.98f, 0.62f, 0.15f, 1.0f}
                : meter_kind_ == MeterKind::Second                 ? glm::vec4{0.25f, 0.55f, 0.98f, 1.0f}
                                                                : glm::vec4{0.35f, 0.85f, 0.45f, 1.0f};
        list.fill_rect({content.x, content.y + content.h - kMeterHeight, width, kMeterHeight}, color);
    }
    if (selected_) {
        list.stroke_rect(content, {1.0f, 0.95f, 0.3f, 1.0f}, 2.0f);
    }
}

void FilterPaint::show(CellImage image) {
    image_ = std::move(image);
    ++version_;
}

void FilterPaint::clear() {
    image_ = CellImage{};
    ++version_;
}

void FilterPaint::vacate() {
    vacant_ = true;
    ++version_;
}

void FilterPaint::set_selected(bool selected) {
    if (selected_ != selected) {
        selected_ = selected;
        ++version_;
    }
}

void FilterPaint::set_meter(float meter, MeterKind kind) {
    const int steps = meter < 0.0f ? -1 : static_cast<int>(std::lround(std::min(meter, 1.0f) * kMeterSteps));
    if (meter_steps_ != steps || (steps > 0 && meter_kind_ != kind)) {
        meter_steps_ = steps;
        meter_kind_ = kind;
        ++version_;
    }
}

}
