#pragma once

#include "model/labeled_images.h"

#include <engine/ui/view_model.h>

#include <array>
#include <limits>
#include <memory>
#include <span>

namespace game {

// One thumbnail of the dataset grid: a sample shrunk to kThumbnailSide x kThumbnailSide, a colored mark for its species
// in the corner, a frame when picked.
class GridCell final : public engine::ui::IPaint {
public:
    static constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
    static constexpr int kThumbnailSide = 16;

    void paint(engine::ui::IDrawList& list, const engine::render::Rect& content) override;
    [[nodiscard]] std::uint64_t version() const noexcept override { return version_; }

    // `samples` must outlive the cell (the row keeps it alive). `index` is kNone for an unused cell at the row's end.
    void show(const LabeledImages* samples, std::size_t index);
    void set_selected(bool selected);

private:
    const LabeledImages* samples_ = nullptr;
    std::size_t index_ = kNone;
    bool selected_ = false;
    std::uint64_t version_ = 0;
};

// One row of the grid, kCells samples wide. The grid is an ItemsControl of these, so only the rows on screen are ever
// built into elements, however many samples the dataset has.
class GridRow final : public engine::ui::ViewModel {
public:
    static constexpr std::size_t kCells = 12;

    // Shows the samples of `samples` listed in `indices` (at most kCells of them, left to right); the cells after them
    // stay empty and cannot be picked.
    GridRow(std::shared_ptr<const LabeledImages> samples, std::span<const std::size_t> indices);

    [[nodiscard]] GridCell& cell(std::size_t column) { return cells_[column]; }
    [[nodiscard]] engine::ui::RelayCommand& pick(std::size_t column) { return picks_[column]; }

private:
    std::shared_ptr<const LabeledImages> samples_;
    std::array<GridCell, kCells> cells_;
    std::array<engine::ui::RelayCommand, kCells> picks_;
};

}
