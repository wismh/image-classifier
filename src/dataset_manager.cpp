#include "dataset_manager.h"

#include "model/path_display.h"

#include <algorithm>

namespace game {

DatasetManager::DatasetManager(PetsViewModel& view, DatasetStore store, const Tr& tr, StatusLine& status,
        engine::IWindowControl& windows)
    : view_(view)
    , store_(std::move(store))
    , tr_(tr)
    , status_(status)
    , dialog_(windows)
    , list_(view.datasets, [this](std::size_t index) { select_dataset(list_.names()[index]); })
    , samples_(std::make_shared<const LabeledImages>()) {
    view_.browseFile.bind_to<DatasetManager, &DatasetManager::browse_file, &DatasetManager::can_browse>(*this);
    view_.addDataset.bind_to<DatasetManager, &DatasetManager::add_dataset>(*this);
    view_.deleteDataset.bind_to<DatasetManager, &DatasetManager::delete_dataset, &DatasetManager::can_delete_dataset>(
            *this);
    view_.pagePrev.bind_to<DatasetManager, &DatasetManager::page_prev, &DatasetManager::can_page_prev>(*this);
    view_.pageNext.bind_to<DatasetManager, &DatasetManager::page_next, &DatasetManager::can_page_next>(*this);

    show_file();
    refresh_list({});
}

void DatasetManager::apply_locale() {
    show_file();
    show_texts();
}

void DatasetManager::update() {
    if (const auto chosen = dialog_.take()) {
        offer_file(*chosen);
    }
}

void DatasetManager::browse_file() {
    dialog_.choose_file({engine::FileFilter{tr_("filter.datasets"), "cvds"},
            engine::FileFilter{tr_("filter.all"), "*"}});
}

void DatasetManager::offer_file(const std::filesystem::path& file) {
    file_ = file;
    view_.addName = DatasetStore::suggest_name(file);
    show_file();
}

void DatasetManager::show_file() {
    view_.addFileText = file_.empty() ? tr_("dialog.none") : display_path(file_, kShownPathLength);
}

void DatasetManager::add_dataset() {
    const std::filesystem::path file = file_;
    if (file.empty()) {
        status_.say("status.file_missing");
        return;
    }
    std::string name = view_.addName.get();
    if (name.empty()) {
        name = DatasetStore::suggest_name(file);
    }
    if (status_.reject_name(DatasetStore::check_name(name))) {
        return;
    }
    if (store_.contains(name)) {
        status_.say("status.dataset_exists", {{"name", name}});
        return;
    }
    const auto added = store_.add_file(file, name);
    if (!added) {
        status_.say("status.add_failed", {{"detail", added.error()}});
        return;
    }
    view_.addName = std::string{};
    refresh_list(name);
    status_.say("status.dataset_added", {{"name", name}, {"count", std::to_string(*added)}});
}

void DatasetManager::delete_dataset() {
    const std::string name = list_.selected_name();
    if (!store_.remove(name)) {
        status_.say("status.dataset_delete_failed", {{"name", name}});
        return;
    }
    refresh_list({});
    status_.say("status.dataset_deleted", {{"name", name}});
}

void DatasetManager::select_dataset(const std::string& name) {
    if (const auto index = list_.find(name)) {
        list_.select(*index);
    } else {
        list_.clear_selection();
    }
    selected_sample_.reset();
    page_ = 0;
    view_.samplePaint.clear();

    samples_ = std::make_shared<const LabeledImages>();
    if (has_active()) {
        auto loaded = store_.load(name);
        if (loaded) {
            samples_ = std::make_shared<const LabeledImages>(std::move(*loaded));
        } else {
            status_.say("status.load_failed", {{"detail", loaded.error()}});
        }
    }
    order_ = SampleOrder{*samples_};
    rebuild_grid();
    show_texts();
}

std::optional<DatasetManager::SelectedSample> DatasetManager::selected_sample() const {
    if (!selected_sample_ || *selected_sample_ >= samples_->size()) {
        return std::nullopt;
    }
    return SelectedSample{samples_->image(*selected_sample_),
            static_cast<ClassId>(samples_->labels[*selected_sample_]), *selected_sample_ + 1};
}

void DatasetManager::page_prev() {
    --page_;
    rebuild_grid();
    show_texts();
}

void DatasetManager::page_next() {
    ++page_;
    rebuild_grid();
    show_texts();
}

void DatasetManager::select_sample(std::size_t index) {
    const auto cell_of = [this](std::size_t sample) -> GridCell* {
        const std::size_t position = order_.position_of(sample);
        if (position < page_first() || position >= page_first() + kPageSamples) {
            return nullptr;
        }
        const std::size_t offset = position - page_first();
        return &rows_[offset / GridRow::kCells]->cell(offset % GridRow::kCells);
    };
    if (selected_sample_) {
        if (GridCell* cell = cell_of(*selected_sample_)) {
            cell->set_selected(false);
        }
    }
    selected_sample_ = index;
    if (GridCell* cell = cell_of(index)) {
        cell->set_selected(true);
    }
    view_.samplePaint.show(samples_->image(index));
    show_texts();
}

void DatasetManager::refresh_list(const std::string& select_name) {
    list_.reset(store_.list());
    const auto& names = list_.names();
    if (names.empty()) {
        select_dataset({});
    } else {
        select_dataset(list_.contains(select_name) ? select_name : names.front());
    }
}

void DatasetManager::rebuild_grid() {
    rows_.clear();
    const std::size_t end = std::min(samples_->size(), page_first() + kPageSamples);
    for (std::size_t first = page_first(); first < end; first += GridRow::kCells) {
        std::vector<std::size_t> indices;
        for (std::size_t position = first; position < std::min(end, first + GridRow::kCells); ++position) {
            indices.push_back(order_.sample_at(position));
        }
        auto row = std::make_shared<GridRow>(samples_, indices);
        for (std::size_t column = 0; column < indices.size(); ++column) {
            row->pick(column) = [this, index = indices[column]] { select_sample(index); };
        }
        rows_.push_back(std::move(row));
    }
    if (selected_sample_) {
        const std::size_t position = order_.position_of(*selected_sample_);
        if (position >= page_first() && position < end) {
            const std::size_t offset = position - page_first();
            rows_[offset / GridRow::kCells]->cell(offset % GridRow::kCells).set_selected(true);
        }
    }
    view_.gridRows.set(rows_);
}

void DatasetManager::show_texts() {
    const std::size_t count = samples_->size();
    const std::string total = std::to_string(count);
    view_.datasetTitleText = has_active() ? tr_("ds.title", {{"name", list_.selected_name()}, {"count", total}})
                                     : tr_("ds.title_none");
    const ClassNames& names = samples_->class_names;
    view_.datasetCountsText = tr_("ds.counts",
            {{"first", names[0]}, {"first_count", std::to_string(order_.first_count())}, {"second", names[1]},
                    {"second_count", std::to_string(order_.second_count())}});
    view_.pageText = tr_("ds.page",
            {{"first", std::to_string(count == 0 ? 0 : page_first() + 1)},
                    {"last", std::to_string(std::min(count, page_first() + kPageSamples))}, {"total", total}});
    if (selected_sample_ && *selected_sample_ < count) {
        view_.sampleInfoText = tr_("ds.sample", {{"n", std::to_string(*selected_sample_ + 1)},
                {"class", names[samples_->labels[*selected_sample_]]}});
    } else {
        view_.sampleInfoText = tr_("ds.no_sample");
    }
}

}
