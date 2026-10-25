#pragma once

#include "loc/localized.h"
#include "loc/tr.h"
#include "model/dataset_store.h"
#include "model/labeled_images.h"
#include "model/sample_order.h"
#include "path_dialog.h"
#include "status_line.h"
#include "view/name_list.h"
#include "view/pets_view_model.h"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace game {

// The normalized datasets of the app and the one on show (step 2 of the Datasets tab adds to them). The one on show is
// listed as a grid of thumbnails, the first class first and then the second.
class DatasetManager final : public Localized {
public:
    // The grid shows one page of this many samples (a multiple of GridRow::kCells). The UI's cost of a list grows
    // faster than its length: 500 rows open at once, 2 000 take a second. 6 000 samples are 500 rows.
    static constexpr std::size_t kPageSamples = 6000;
    // A path longer than this is shortened on its label.
    static constexpr std::size_t kShownPathLength = 24;

    DatasetManager(PetsViewModel& view, DatasetStore store, const Tr& tr, StatusLine& status,
            engine::IWindowControl& windows);

    DatasetManager(const DatasetManager&) = delete;
    DatasetManager& operator=(const DatasetManager&) = delete;

    // Writes the texts of the Datasets tab again after a change of language.
    void apply_locale() override;
    // Once per frame: takes the file chosen in the dialog.
    void update();

    // Makes `file` the normalized file step 2 will add, and proposes its name for the dataset.
    void offer_file(const std::filesystem::path& file);

    // The dataset on show: what training runs on. Empty (and no samples) while there is none. The samples are a
    // snapshot that later changes to the list do not touch.
    [[nodiscard]] bool has_active() const { return list_.selected().has_value(); }
    [[nodiscard]] std::string active_name() const { return list_.selected_name(); }
    [[nodiscard]] std::shared_ptr<const LabeledImages> active_samples() const { return samples_; }

    // The sample picked in the grid. `picture` points into the dataset on show and holds until the list or the dataset
    // changes; copy it to keep it.
    struct SelectedSample {
        std::span<const std::uint8_t> picture;
        ClassId species;
        std::size_t number;  // 1-based, in the order the dataset file lists them
    };
    [[nodiscard]] std::optional<SelectedSample> selected_sample() const;

private:
    using Rows = std::vector<std::shared_ptr<GridRow>>;

    [[nodiscard]] std::size_t page_count() const { return order_.page_count(kPageSamples); }
    [[nodiscard]] std::size_t page_first() const { return page_ * kPageSamples; }
    void browse_file();
    [[nodiscard]] bool can_browse() const { return !dialog_.pending(); }
    void show_file();
    void add_dataset();
    void delete_dataset();
    [[nodiscard]] bool can_delete_dataset() const { return has_active(); }
    void page_prev();
    void page_next();
    [[nodiscard]] bool can_page_prev() const { return page_ > 0; }
    [[nodiscard]] bool can_page_next() const { return page_ + 1 < page_count(); }
    void select_dataset(const std::string& name);
    void select_sample(std::size_t index);

    void refresh_list(const std::string& select_name);
    void rebuild_grid();
    void show_texts();

    PetsViewModel& view_;
    DatasetStore store_;
    const Tr& tr_;
    StatusLine& status_;
    PathDialog dialog_;
    std::filesystem::path file_;  // the normalized file step 2 adds

    NameList list_;  // the datasets; the selected one is on show
    std::shared_ptr<const LabeledImages> samples_;
    SampleOrder order_{LabeledImages{}};  // which sample is shown at which position of the grid
    std::size_t page_ = 0;  // a page is kPageSamples positions of the grid
    Rows rows_;             // the rows of the page shown
    std::optional<std::size_t> selected_sample_;
};

}
