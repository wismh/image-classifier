#pragma once

#include "model/pet_image.h"

#include <engine/ui/paint.h>

#include <cstdint>
#include <span>
#include <vector>

namespace game {

// One normalized picture drawn pixel by pixel into whatever element it is bound to: the preview of the picked sample,
// the picture the network looks at. Without a picture it shows an empty frame. The paint keeps its own copy.
class SamplePaint final : public engine::ui::IPaint {
public:
    void paint(engine::ui::IDrawList& list, const engine::render::Rect& content) override;
    [[nodiscard]] std::uint64_t version() const noexcept override { return version_; }

    // `picture` is kImageBytes bytes.
    void show(std::span<const std::uint8_t> picture);
    void clear();

private:
    std::vector<std::uint8_t> picture_;
    std::uint64_t version_ = 0;
};

}
