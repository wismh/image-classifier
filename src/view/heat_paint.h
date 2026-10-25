#pragma once

#include "model/pet_image.h"

#include <engine/ui/paint.h>

#include <cstdint>
#include <span>
#include <vector>

namespace game {

// A picture with a heat map over it (Grad-CAM): the places the map is high are tinted from yellow to red, the rest is
// the
// picture a little darker. The map is smaller than the picture and is stretched over it smoothly. Without a picture it
// shows an empty frame, without a map the picture alone. The paint keeps its own copies.
class HeatPaint final : public engine::ui::IPaint {
public:
    void paint(engine::ui::IDrawList& list, const engine::render::Rect& content) override;
    [[nodiscard]] std::uint64_t version() const noexcept override { return version_; }

    // `picture` is kImageBytes bytes; `map` is side x side values in 0..1.
    void show(std::span<const std::uint8_t> picture, std::span<const float> map, int side);
    void show_picture(std::span<const std::uint8_t> picture);
    void clear();

private:
    // The map at the middle of the picture's pixel (x, y), by bilinear interpolation between the centers of its cells.
    [[nodiscard]] float heat_at(int x, int y) const;

    std::vector<std::uint8_t> picture_;
    std::vector<float> map_;
    int side_ = 0;
    std::uint64_t version_ = 0;
};

}
