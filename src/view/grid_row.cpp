#include "view/grid_row.h"

#include <format>

namespace game {

void GridCell::paint(engine::ui::IDrawList& list, const engine::render::Rect& content) {
    if (index_ == kNone) {
        return;
    }

    constexpr int kBlock = kImageSide / kThumbnailSide;
    const float pixel = content.w / kThumbnailSide;
    const std::span<const std::uint8_t> image = samples_->image(index_);
    for (int ty = 0; ty < kThumbnailSide; ++ty) {
        for (int tx = 0; tx < kThumbnailSide; ++tx) {
            int sum[kChannels]{};
            for (int y = ty * kBlock; y < (ty + 1) * kBlock; ++y) {
                for (int x = tx * kBlock; x < (tx + 1) * kBlock; ++x) {
                    const std::uint8_t* rgb = image.data() + (static_cast<std::size_t>(y) * kImageSide + x) * kChannels;
                    for (int c = 0; c < kChannels; ++c) {
                        sum[c] += rgb[c];
                    }
                }
            }
            constexpr float kScale = 1.0f / (255.0f * kBlock * kBlock);
            list.fill_rect({content.x + tx * pixel, content.y + ty * pixel, pixel + 0.5f, pixel + 0.5f},
                    {sum[0] * kScale, sum[1] * kScale, sum[2] * kScale, 1.0f});
        }
    }

    const bool first = samples_->labels[index_] == static_cast<std::uint8_t>(ClassId::First);
    list.fill_rect({content.x, content.y + content.h - 4.0f, content.w, 4.0f},
            first ? glm::vec4{0.98f, 0.62f, 0.15f, 1.0f} : glm::vec4{0.25f, 0.55f, 0.98f, 1.0f});
    if (selected_) {
        list.stroke_rect(content, {1.0f, 0.95f, 0.3f, 1.0f}, 2.0f);
    }
}

void GridCell::show(const LabeledImages* samples, std::size_t index) {
    samples_ = samples;
    index_ = index;
    ++version_;
}

void GridCell::set_selected(bool selected) {
    if (selected_ != selected) {
        selected_ = selected;
        ++version_;
    }
}

namespace {

// "cell0".."cell11" and "pick0".."pick11", hashed once: a big dataset builds thousands of rows.
struct ColumnIds {
    std::array<engine::ui::BindingId, GridRow::kCells> cells;
    std::array<engine::ui::BindingId, GridRow::kCells> picks;

    ColumnIds() {
        for (std::size_t column = 0; column < GridRow::kCells; ++column) {
            cells[column] = engine::ui::intern(std::format("cell{}", column));
            picks[column] = engine::ui::intern(std::format("pick{}", column));
        }
    }
};

const ColumnIds& column_ids() {
    static const ColumnIds ids;
    return ids;
}

}

GridRow::GridRow(std::shared_ptr<const LabeledImages> samples, std::span<const std::size_t> indices)
    : samples_(std::move(samples)) {
    for (std::size_t column = 0; column < kCells; ++column) {
        const bool used = column < indices.size();
        cells_[column].show(samples_.get(), used ? indices[column] : GridCell::kNone);
        picks_[column].set_can_execute(used);

        paint(column_ids().cells[column], cells_[column]);
        command(column_ids().picks[column], picks_[column]);
    }
}

}
