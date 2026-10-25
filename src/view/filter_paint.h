#pragma once

#include "model/cell_image.h"

#include <engine/ui/paint.h>

namespace game {

// What the bar at the foot of a cell stands for: how strong a filter's weights are (green), or a push toward the first
// class (orange)
// or the second (blue).
enum class MeterKind { Strength, First, Second };

// One CellImage drawn into whatever element it is bound to, pixel by pixel: a cell of the filter grid, or one of the
// big
// pictures beside it. A cell can also carry a bar (how strong the filter is) and a frame when it is picked. Without an
// image it is an empty frame.
class FilterPaint final : public engine::ui::IPaint {
public:
    void paint(engine::ui::IDrawList& list, const engine::render::Rect& content) override;
    [[nodiscard]] std::uint64_t version() const noexcept override { return version_; }

    void show(CellImage image);
    void clear();
    // Paints nothing at all: a cell at the end of a row that has no unit.
    void vacate();
    void set_selected(bool selected);
    // 0..1, or negative for no bar. Quantized, so a small change does not redraw the picture.
    void set_meter(float meter, MeterKind kind = MeterKind::Strength);

private:
    CellImage image_;
    bool selected_ = false;
    bool vacant_ = false;
    int meter_steps_ = -1;
    MeterKind meter_kind_ = MeterKind::Strength;
    std::uint64_t version_ = 0;
};

}
