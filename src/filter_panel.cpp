#include "filter_panel.h"

#include <algorithm>
#include <cmath>
#include <format>

namespace game {

namespace {

// Pictures in a cell are drawn as at most this many pixels a side: a cell is about 60 pixels wide, and the grid holds
// up to
// 64 of them, so a full 64 x 64 picture in each would be thousands of rectangles for nothing.
constexpr int kCellPixels = 32;

float largest_size(const std::vector<float>& values) {
    float largest = 0.0f;
    for (const float value : values) {
        largest = std::max(largest, std::abs(value));
    }
    return largest;
}

}

FilterPanel::FilterPanel(PetsViewModel& view, const Tr& tr)
    : view_(view)
    , tr_(tr) {
    view_.filterBlockUp.bind_to<FilterPanel, &FilterPanel::layer_up, &FilterPanel::can_layer_up>(*this);
    view_.filterBlockDown.bind_to<FilterPanel, &FilterPanel::layer_down, &FilterPanel::can_layer_down>(*this);
    view_.showFeatures.bind_to<FilterPanel, &FilterPanel::show_answers, &FilterPanel::has_model>(*this);
    view_.showWeights.bind_to<FilterPanel, &FilterPanel::show_weights, &FilterPanel::has_model>(*this);
    view_.showDream.bind_to<FilterPanel, &FilterPanel::show_dream, &FilterPanel::has_model>(*this);
    view_.filtersReady = false;
    view_.filtersMissing = true;
    show_texts();
}

void FilterPanel::set_model(std::shared_ptr<const Cnn> model, ClassNames names) {
    model_ = std::move(model);
    names_ = std::move(names);
    dreamer_.reset(model_);
    view_.filtersReady = model_ != nullptr;
    view_.filtersMissing = model_ == nullptr;
    if (!model_) {
        return;
    }
    workspace_ = model_->make_workspace();
    if (picture_) {
        votes_ = run_filters(*model_, workspace_, *picture_);
    }
    enter_layer(0);
}

void FilterPanel::set_picture(std::span<const std::uint8_t> picture) {
    picture_.emplace(picture.begin(), picture.end());
    view_.filterThumbPaint.show(picture);
    if (!model_) {
        return;
    }
    votes_ = run_filters(*model_, workspace_, *picture_);
    dirty_ = true;
}

void FilterPanel::update(bool visible) {
    const bool arrived = dreamer_.collect();
    if (!model_ || !visible) {
        dirty_ = dirty_ || arrived;
        return;
    }
    if (arrived || dirty_) {
        refresh();
    }
    dirty_ = false;
}

void FilterPanel::refresh() {
    fill_cells();
    show_detail();
    show_votes();
    show_texts();
}

void FilterPanel::apply_locale() {
    show_texts();
    if (model_) {
        show_votes();
    }
}

void FilterPanel::layer_up() {
    enter_layer(layer_ + 1);
}

void FilterPanel::layer_down() {
    enter_layer(layer_ - 1);
}

void FilterPanel::set_mode(Mode mode) {
    mode_ = mode;
    if (mode_ == Mode::Dream) {
        dreamer_.ask(layer_);
    }
    refresh();
}

void FilterPanel::select(int unit) {
    cell(selected_).set_selected(false);
    selected_ = unit;
    cell(selected_).set_selected(true);
    show_detail();
    show_votes();
    show_texts();
}

void FilterPanel::enter_layer(int layer) {
    layer_ = layer;
    selected_ = 0;
    weights_.clear();
    for (int unit = 0; unit < unit_count(); ++unit) {
        weights_.push_back(weights_of(unit));
    }
    if (mode_ == Mode::Dream) {
        dreamer_.ask(layer_);
    }
    build_grid();
    refresh();
}

CellImage FilterPanel::weights_of(int unit) const {
    if (FilterLayers::is_conv(layer_)) {
        return downsampled(weights_image(filter_weights_picture(*model_, layer_, unit)), kCellPixels);
    }
    return weights_image(is_hidden() ? hidden_weights_picture(*model_, unit)
                                     : output_weights_picture(*model_, static_cast<ClassId>(unit)));
}

void FilterPanel::build_grid() {
    rows_.clear();
    const auto count = static_cast<std::size_t>(unit_count());
    for (std::size_t first = 0; first < count; first += FilterRow::kCells) {
        auto row = std::make_shared<FilterRow>(std::min(FilterRow::kCells, count - first));
        for (std::size_t column = 0; column < row->used(); ++column) {
            row->pick(column) = [this, unit = static_cast<int>(first + column)] { select(unit); };
        }
        rows_.push_back(std::move(row));
    }
    view_.filterRows.set(rows_);
}

FilterPaint& FilterPanel::cell(int unit) {
    const auto index = static_cast<std::size_t>(unit);
    return rows_[index / FilterRow::kCells]->cell(index % FilterRow::kCells);
}

float FilterPanel::answer_scale() const {
    if (!picture_ || is_output()) {
        return 0.0f;
    }
    float largest = 0.0f;
    for (int unit = 0; unit < unit_count(); ++unit) {
        if (FilterLayers::is_conv(layer_)) {
            const auto map = activation_map(workspace_, layer_, unit);
            largest = std::max(largest, *std::ranges::max_element(map));
        } else {
            largest = std::max(largest, hidden_activation(workspace_, unit));
        }
    }
    return largest > 0.0f ? 1.0f / largest : 0.0f;
}

CellImage FilterPanel::answer_image(int unit, float scale, bool full) const {
    if (!picture_) {
        return {};
    }
    if (FilterLayers::is_conv(layer_)) {
        const auto map = activation_map(workspace_, layer_, unit);
        if (full) {
            // The big picture is stretched to the full range on its own, so that the shape shows even when it is weak.
            const float largest = *std::ranges::max_element(map);
            return gray_image(map, Cnn::block_side(layer_), largest > 0.0f ? 1.0f / largest : 0.0f);
        }
        return downsampled(gray_image(map, Cnn::block_side(layer_), scale), kCellPixels);
    }
    // A neuron or an output answers with one number: a single gray tile as bright as the answer.
    const float value = is_hidden() ? hidden_activation(workspace_, unit)
                                    : class_probability(workspace_, static_cast<ClassId>(unit));
    return gray_image(std::vector<float>{value}, 1, is_output() ? 1.0f : scale);
}

std::vector<float> FilterPanel::votes() const {
    if (!picture_ || is_output()) {
        return {};
    }
    return FilterLayers::is_conv(layer_) ? votes_.conv[static_cast<std::size_t>(layer_)] : votes_.hidden;
}

std::vector<float> FilterPanel::connections() const {
    std::vector<float> connections;
    for (int neuron = 0; neuron < model_->hidden_neurons(); ++neuron) {
        connections.push_back(hidden_connection(*model_, neuron));
    }
    return connections;
}

FilterPanel::Meter FilterPanel::meter(int unit) const {
    const auto toward = [](float value, float largest) {
        return Meter{largest > 0.0f ? std::abs(value) / largest : 0.0f,
                value > 0.0f ? MeterKind::Second : MeterKind::First};
    };
    switch (mode_) {
    case Mode::Answers: {
        const std::vector<float> pushed = votes();
        return pushed.empty() ? Meter{} : toward(pushed[static_cast<std::size_t>(unit)], largest_size(pushed));
    }
    case Mode::Weights:
    case Mode::Dream:
        if (is_hidden()) {
            // What does not change with the picture: how the output layer takes the neuron.
            const std::vector<float> fixed = connections();
            return toward(fixed[static_cast<std::size_t>(unit)], largest_size(fixed));
        }
        if (mode_ == Mode::Weights && FilterLayers::is_conv(layer_)) {
            return {filter_strength(*model_, layer_, unit), MeterKind::Strength};
        }
        return {};
    }
    return {};
}

void FilterPanel::fill_cells() {
    const float scale = mode_ == Mode::Answers ? answer_scale() : 0.0f;
    for (int unit = 0; unit < unit_count(); ++unit) {
        FilterPaint& paint = cell(unit);
        switch (mode_) {
        case Mode::Answers: paint.show(answer_image(unit, scale, false)); break;
        case Mode::Weights: paint.show(weights_[static_cast<std::size_t>(unit)]); break;
        case Mode::Dream: {
            const auto& dream = dreamer_.picture(layer_, unit);
            paint.show(dream ? downsampled(color_image(*dream, kImageSide), kCellPixels) : CellImage{});
            break;
        }
        }
        const Meter bar = meter(unit);
        paint.set_meter(bar.size, bar.kind);
        paint.set_selected(unit == selected_);
    }
    view_.featuresActive = mode_ == Mode::Answers;
    view_.featuresIdle = mode_ != Mode::Answers;
    view_.weightsActive = mode_ == Mode::Weights;
    view_.weightsIdle = mode_ != Mode::Weights;
    view_.dreamActive = mode_ == Mode::Dream;
    view_.dreamIdle = mode_ != Mode::Dream;
}

void FilterPanel::show_detail() {
    const CellImage answer = answer_image(selected_, answer_scale(), true);
    if (answer.kind == CellImage::Kind::None) {
        view_.filterMapPaint.clear();
    } else {
        view_.filterMapPaint.show(answer);
    }
    view_.filterWeightsPaint.show(FilterLayers::is_conv(layer_)
                    ? weights_image(filter_weights_picture(*model_, layer_, selected_))
                    : weights_[static_cast<std::size_t>(selected_)]);
    const auto& dream = dreamer_.picture(layer_, selected_);
    if (dream) {
        view_.filterDreamPaint.show(color_image(*dream, kImageSide));
    } else {
        view_.filterDreamPaint.clear();
    }
}

void FilterPanel::show_votes() {
    // The chart is what the picture made of each unit; without a picture a neuron still has its fixed connection.
    std::vector<float> pushed = votes();
    if (pushed.empty() && is_hidden()) {
        pushed = connections();
    }
    if (pushed.empty()) {
        view_.filterVotesPaint.clear();
    } else {
        view_.filterVotesPaint.show(pushed[static_cast<std::size_t>(selected_)], largest_size(pushed));
    }
    view_.filterVotesPaint.set_names(names_[0], names_[1]);
}

void FilterPanel::show_texts() {
    if (!model_) {
        view_.filterLayerText = tr_("filters.no_model");
        view_.filterTitleText = std::string{};
        view_.filterDetailText = std::string{};
        view_.filterNoteText = std::string{};
        view_.filterProgressText = std::string{};
        return;
    }
    const int count = unit_count();
    if (FilterLayers::is_conv(layer_)) {
        const int side = Cnn::block_side(layer_);
        view_.filterLayerText = tr_("filters.layer", {{"n", std::to_string(layer_ + 1)},
                {"count", std::to_string(FilterLayers::kCount)}, {"filters", std::to_string(count)},
                {"side", std::to_string(side)}});
    } else {
        view_.filterLayerText = tr_(is_hidden() ? "filters.layer_hidden" : "filters.layer_output",
                {{"n", std::to_string(layer_ + 1)}, {"count", std::to_string(FilterLayers::kCount)},
                        {"units", std::to_string(count)}});
    }

    // The picked unit.
    const std::string number = std::to_string(selected_ + 1);
    const std::string total = std::to_string(count);
    if (is_output()) {
        view_.filterTitleText = tr_("filters.selected_output",
                {{"class", names_[static_cast<std::size_t>(selected_)]}});
        const std::string chance = picture_
                ? std::format("{:.0f}", class_probability(workspace_, static_cast<ClassId>(selected_)) * 100.0f)
                : std::string{"-"};
        view_.filterDetailText = tr_("filters.detail_output",
                {{"bias", std::format("{:+.3f}", model_->output_bias(selected_))}, {"chance", chance}});
    } else {
        view_.filterTitleText = tr_(is_hidden() ? "filters.selected_neuron" : "filters.selected",
                {{"n", number}, {"count", total}});
        const float bias = is_hidden() ? model_->hidden_bias(selected_) : model_->conv_bias(layer_, selected_);
        const float strength =
                is_hidden() ? hidden_strength(*model_, selected_) : filter_strength(*model_, layer_, selected_);
        std::string answer = "-";
        if (picture_) {
            const float activation = is_hidden() ? hidden_activation(workspace_, selected_)
                                                 : mean_activation(activation_map(workspace_, layer_, selected_));
            answer = std::format("{:.3f}", activation);
        }
        std::string text = tr_("filters.detail", {{"bias", std::format("{:+.3f}", bias)},
                {"strength", std::format("{:.0f}", strength * 100.0f)}, {"answer", answer}});
        const auto say_push = [&](const char* key, float value) {
            text += "\n" + tr_(key, {{"class", names_[value > 0.0f ? 1 : 0]},
                                          {"value", std::format("{:.3f}", std::abs(value))}});
        };
        const std::vector<float> pushed = votes();
        if (!pushed.empty()) {
            say_push("filters.vote", pushed[static_cast<std::size_t>(selected_)]);
        } else {
            text += "\n" + tr_("filters.vote_none");
        }
        if (is_hidden()) {
            say_push("filters.connection", hidden_connection(*model_, selected_));
        }
        view_.filterDetailText = text;
    }

    view_.filterNoteText = mode_ == Mode::Answers && !picture_ ? tr_("filters.note_no_picture") : std::string{};
    if (mode_ == Mode::Dream && dreamer_.asked(layer_) && dreamer_.done(layer_) < dreamer_.total(layer_)) {
        view_.filterProgressText = tr_("filters.progress", {{"done", std::to_string(dreamer_.done(layer_))},
                {"total", std::to_string(dreamer_.total(layer_))}});
    } else {
        view_.filterProgressText = std::string{};
    }
}

}
