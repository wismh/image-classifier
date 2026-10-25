#pragma once

#include <engine/ui/paint.h>

#include <string>

namespace game {

// Two horizontal bars, one for each class, showing which way the picked unit pushes: the bar of its side is
// as long as its push is next to the strongest push of the layer, the other stays empty. Empty while there is nothing
// to
// show.
class VotesPaint final : public engine::ui::IPaint {
public:
    void paint(engine::ui::IDrawList& list, const engine::render::Rect& content) override;
    [[nodiscard]] std::uint64_t version() const noexcept override { return version_; }

    // `vote` is positive for the second class and negative for the first; `strongest` is the largest size of a vote in
    // the layer.
    void show(float vote, float strongest);
    void clear();
    void set_names(std::string first, std::string second);

private:
    bool present_ = false;
    float vote_ = 0.0f;
    float strongest_ = 0.0f;
    std::string first_name_ = "class 1";
    std::string second_name_ = "class 2";
    std::uint64_t version_ = 0;
};

}
