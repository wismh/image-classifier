#pragma once

#include <engine/ui/paint.h>

#include <string>

namespace game {

// Two horizontal bars, the probability of each class, each with its name and percentage. Without a prediction
// the bars are empty.
class ProbabilityPaint final : public engine::ui::IPaint {
public:
    void paint(engine::ui::IDrawList& list, const engine::render::Rect& content) override;
    [[nodiscard]] std::uint64_t version() const noexcept override { return version_; }

    // Shares 0..1; quantized to whole percents, so a change too small to see does not redraw.
    void show(float first, float second);
    void clear();
    // The names beside the bars, in the language on screen.
    void set_names(std::string first, std::string second);

private:
    bool present_ = false;
    int first_percent_ = 0;
    int second_percent_ = 0;
    std::string first_name_ = "Class 1";
    std::string second_name_ = "Class 2";
    std::uint64_t version_ = 0;
};

}
