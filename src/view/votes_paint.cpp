#include "view/votes_paint.h"

#include <engine/builtin_ids.h>

#include <algorithm>
#include <cmath>

namespace game {

namespace {

constexpr float kNameWidth = 64.0f;
constexpr float kBarHeight = 16.0f;

}

void VotesPaint::paint(engine::ui::IDrawList& list, const engine::render::Rect& content) {
    list.fill_rect(content, {0.97f, 0.98f, 0.99f, 1.0f});
    const float share = present_ && strongest_ > 0.0f ? std::min(1.0f, std::abs(vote_) / strongest_) : 0.0f;
    const float reach = content.w - kNameWidth - 8.0f;
    const struct {
        const std::string* name;
        glm::vec4 color;
        bool active;
    } rows[2] = {
        {&first_name_, {0.98f, 0.62f, 0.15f, 1.0f}, present_ && vote_ < 0.0f},
        {&second_name_, {0.25f, 0.55f, 0.98f, 1.0f}, present_ && vote_ > 0.0f},
    };
    list.set_font(engine::builtin::font_ui, 12.0f);
    for (int i = 0; i < 2; ++i) {
        const float top = content.y + 8.0f + static_cast<float>(i) * (kBarHeight + 8.0f);
        list.text(*rows[i].name, {content.x + 6.0f, top + 1.0f}, {0.2f, 0.25f, 0.33f, 1.0f});
        list.fill_rect({content.x + kNameWidth, top, reach, kBarHeight}, {0.89f, 0.91f, 0.94f, 1.0f});
        if (rows[i].active) {
            list.fill_rect({content.x + kNameWidth, top, std::max(2.0f, reach * share), kBarHeight}, rows[i].color);
        }
    }
}

void VotesPaint::show(float vote, float strongest) {
    present_ = true;
    vote_ = vote;
    strongest_ = strongest;
    ++version_;
}

void VotesPaint::clear() {
    present_ = false;
    ++version_;
}

void VotesPaint::set_names(std::string first, std::string second) {
    first_name_ = std::move(first);
    second_name_ = std::move(second);
    ++version_;
}

}
