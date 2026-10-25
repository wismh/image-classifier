#pragma once

#include "model/pet_image.h"

#include <engine/ui/paint.h>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace game {

// The ideal picture of each class side by side, each under its name: the pictures the network is surest about. A
// picture
// that is not ready yet is an empty frame.
class IdealPaint final : public engine::ui::IPaint {
public:
    static constexpr float kGap = 16.0f;
    static constexpr float kCaptionHeight = 24.0f;

    void paint(engine::ui::IDrawList& list, const engine::render::Rect& content) override;
    [[nodiscard]] std::uint64_t version() const noexcept override { return version_; }

    // `picture` is kImageBytes bytes.
    void set(ClassId species, std::span<const std::uint8_t> picture);
    // Back to empty frames: the network changed.
    void reset();
    void set_names(std::string first, std::string second);

private:
    std::array<std::vector<std::uint8_t>, kNumClasses> pictures_;
    std::array<std::string, kNumClasses> names_{"Class 1", "Class 2"};
    std::uint64_t version_ = 0;
};

}
