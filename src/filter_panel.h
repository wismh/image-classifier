#pragma once

#include "filter_dreamer.h"
#include "loc/localized.h"
#include "loc/tr.h"
#include "model/cell_image.h"
#include "model/cnn.h"
#include "model/filter_view.h"
#include "view/filter_row.h"
#include "view/pets_view_model.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace game {

// The Filters tab: what each unit of the network in use looks for and what it answers. One layer at a time is a grid of
// pictures, one per unit (a filter of a convolution, a neuron of the hidden layer, or one of the two outputs), of one
// of
// three kinds: the unit's answer to the picture the network is looking at, its weights, or its dream, the picture it
// answers to hardest (found in the background). Picking a unit shows all three of it, large, and a chart shows how
// every
// unit of the layer pushed the verdict toward "cat" or "dog".
class FilterPanel final : public Localized {
public:
    FilterPanel(PetsViewModel& view, const Tr& tr);

    FilterPanel(const FilterPanel&) = delete;
    FilterPanel& operator=(const FilterPanel&) = delete;

    // A different network is in use: back to its first layer.
    // `names` are what the network's two classes are called.
    void set_model(std::shared_ptr<const Cnn> model, ClassNames names);
    // The network looks at a new picture (kImageBytes bytes, as in a dataset).
    void set_picture(std::span<const std::uint8_t> picture);
    // Once per frame on the main thread: takes in finished dreams and, while the tab is on screen, shows them.
    void update(bool visible);
    void apply_locale() override;

private:
    enum class Mode { Answers, Weights, Dream };

    [[nodiscard]] bool can_layer_up() const { return model_ && layer_ + 1 < FilterLayers::kCount; }
    [[nodiscard]] bool can_layer_down() const { return model_ && layer_ > 0; }
    [[nodiscard]] bool has_model() const { return model_ != nullptr; }
    [[nodiscard]] int unit_count() const { return FilterLayers::units(*model_, layer_); }
    [[nodiscard]] bool is_hidden() const { return layer_ == FilterLayers::kHidden; }
    [[nodiscard]] bool is_output() const { return layer_ == FilterLayers::kOutput; }

    void layer_up();
    void layer_down();
    void show_answers() { set_mode(Mode::Answers); }
    void show_weights() { set_mode(Mode::Weights); }
    void show_dream() { set_mode(Mode::Dream); }
    void set_mode(Mode mode);
    void select(int unit);

    void enter_layer(int layer);
    void build_grid();
    void fill_cells();
    void show_detail();
    void show_votes();
    void show_texts();
    void refresh();

    // How the unit answered to the picture: a map for a filter, one number for a neuron or an output.
    [[nodiscard]] CellImage answer_image(int unit, float scale, bool full) const;
    [[nodiscard]] CellImage weights_of(int unit) const;
    // The scale that makes the strongest answer of the layer white (0 when there is no picture or nothing answered).
    [[nodiscard]] float answer_scale() const;
    // The votes of the layer's units for the picture on show, positive for dog; empty when there are none.
    [[nodiscard]] std::vector<float> votes() const;
    // A hidden neuron's weight to the output, which votes the same way whatever the picture.
    [[nodiscard]] std::vector<float> connections() const;
    struct Meter {
        float size = -1.0f;
        MeterKind kind = MeterKind::Strength;
    };
    [[nodiscard]] Meter meter(int unit) const;
    [[nodiscard]] FilterPaint& cell(int unit);

    PetsViewModel& view_;
    const Tr& tr_;

    std::shared_ptr<const Cnn> model_;
    ClassNames names_;
    Cnn::Workspace workspace_;
    FilterVotes votes_;
    std::optional<std::vector<std::uint8_t>> picture_;
    int layer_ = 0;
    int selected_ = 0;
    Mode mode_ = Mode::Answers;
    FilterDreamer dreamer_;
    std::vector<std::shared_ptr<FilterRow>> rows_;
    // The weights of every unit of the layer on show, worked out when the layer is entered.
    std::vector<CellImage> weights_;
    bool dirty_ = false;  // the grid or the detail is out of date and the tab is not on screen
};

}
