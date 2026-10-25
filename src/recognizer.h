#pragma once

#include "dataset_manager.h"
#include "filter_panel.h"
#include "ideal_dreamer.h"
#include "loc/localized.h"
#include "loc/tr.h"
#include "model/cnn.h"
#include "model/model_file.h"
#include "model/prediction.h"
#include "model_library.h"
#include "model_sink.h"
#include "path_dialog.h"
#include "status_line.h"
#include "view/pets_view_model.h"

#include <optional>
#include <string>
#include <vector>

namespace game {

// The Recognize tab: holds the network in use and the picture it looks at, and shows what it thinks. The picture comes
// from a file or from the sample picked on the Datasets tab; the network from the list of saved ones or straight from a
// training that just ended.
class Recognizer final : public Localized, public ModelSink {
public:
    Recognizer(PetsViewModel& view, const Tr& tr, StatusLine& status, DatasetManager& datasets, ModelLibrary& library,
            FilterPanel& filters, engine::IWindowControl& windows);

    Recognizer(const Recognizer&) = delete;
    Recognizer& operator=(const Recognizer&) = delete;

    void use_model(const std::string& name, const ModelFile& model) override;
    // Writes the texts of the Recognize tab again after a change of language.
    void apply_locale() override { show_texts(); }
    // Once per frame: takes the photo chosen in the dialog and the ideal pictures that are ready.
    void update();

private:
    // Where the picture on show came from, kept as a message and its arguments so a change of language can say it
    // again.
    struct Source {
        std::string key;
        Tr::Args args;
    };

    void load_model();
    [[nodiscard]] bool can_load_model() const { return library_.has_selection(); }
    void browse_photo();
    [[nodiscard]] bool can_browse() const { return !dialog_.pending(); }
    void open_picture(const std::filesystem::path& path);
    void use_sample();
    [[nodiscard]] bool can_use_sample() const { return datasets_.selected_sample().has_value(); }
    void show_picture(std::vector<std::uint8_t> picture, Source source);
    void predict_picture();
    void show_texts();

    PetsViewModel& view_;
    const Tr& tr_;
    StatusLine& status_;
    DatasetManager& datasets_;
    ModelLibrary& library_;
    FilterPanel& filters_;
    PathDialog dialog_;
    IdealDreamer dreamer_;

    std::shared_ptr<const Cnn> model_;
    Cnn::Workspace workspace_;
    std::string model_name_;
    ModelFile info_;  // the numbers of the file of the network in use (without its parameters)
    std::optional<std::vector<std::uint8_t>> picture_;
    Source source_;
    std::optional<Prediction> prediction_;
};

}
