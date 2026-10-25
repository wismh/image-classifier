#pragma once

#include "dataset_manager.h"
#include "loc/localized.h"
#include "loc/tr.h"
#include "model/model_file.h"
#include "model/training_run.h"
#include "model_library.h"
#include "model_sink.h"
#include "status_line.h"
#include "train_settings.h"
#include "view/pets_view_model.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace game {

// The Train tab: trains a network on the dataset that is on show in the Datasets tab, a slice per frame (update() once
// per frame), lists every epoch as it finishes, and when the run ends saves the network of the best epoch and hands it
// to the Recognize tab. A new network starts from random weights; a saved one can be trained further, with the split of
// the dataset it was trained on, and is saved as a new file beside the old one.
class Trainer final : public Localized {
public:
    // How long one frame may spend training before it hands control back to the UI. A frame lasts this long plus one
    // batch plus drawing, so the window keeps moving; a longer slice trains faster, because the rest of the frame is
    // idle
    // time for the worker threads.
    static constexpr std::chrono::milliseconds kSlice{14};
    // Fewer samples of either class than this and there is nothing to learn from or to check against.
    static constexpr std::size_t kMinPerClass = 10;

    Trainer(PetsViewModel& view, const Tr& tr, StatusLine& status, DatasetManager& datasets, ModelLibrary& library,
            ModelSink& models, Parallel parallel);

    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;

    // Writes the texts of the Train tab again after a change of language.
    void apply_locale() override;
    void update();

private:
    void start();
    void continue_training();
    [[nodiscard]] bool can_continue() const;
    // Begins a run of `config.epochs` epochs of `model`, a network of size `network` that was trained for
    // `epochs_before` epochs already, and says `started_key` on the status line.
    void begin(Cnn model, const CnnConfig& network, const TrainingConfig& config, unsigned epochs_before,
            const std::string& started_key, const std::string& from_name);
    // The samples of the dataset on show, or nothing (with the reason said) when they are too few to train on.
    [[nodiscard]] std::shared_ptr<const LabeledImages> usable_samples();
    // Settles what the network about to be trained is saved as: the name typed on the Train tab, or a made up one when
    // it
    // is empty. False (with the reason said) when the typed name cannot be used.
    [[nodiscard]] bool settle_name();
    void stop();
    [[nodiscard]] bool can_start() const;
    [[nodiscard]] bool can_stop() const { return run_.has_value(); }
    void finish();
    void set_running(bool running);
    void show_dataset();
    void show_progress();
    void show_epochs();
    [[nodiscard]] std::string epoch_text(const EpochStats& stats) const;

    PetsViewModel& view_;
    const Tr& tr_;
    StatusLine& status_;
    DatasetManager& datasets_;
    ModelLibrary& library_;
    ModelSink& models_;
    Parallel parallel_;
    TrainSettings settings_;

    std::optional<TrainingRun> run_;
    CnnConfig running_network_;
    std::size_t train_samples_ = 0;
    std::size_t dataset_samples_ = 0;
    std::uint32_t split_seed_ = 0;
    unsigned epochs_before_ = 0;            // epochs the network had before this run (0 for a new one)
    std::optional<float> accuracy_before_;  // its balanced accuracy, when a saved network is trained further
    std::string continued_name_;
    std::string chosen_name_;  // empty: made up when the run ends
    ClassNames class_names_;   // of the dataset the run trains on
    std::vector<EpochStats> shown_;  // the epochs listed so far, to tell the new ones
    std::vector<std::shared_ptr<ListItem>> rows_;

    // What the dataset line says, to write it again only when it changes.
    bool dataset_line_written_ = false;
    std::string shown_dataset_;
    std::size_t shown_dataset_size_ = 0;
    std::string saved_name_;
};

}
