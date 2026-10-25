#include "trainer.h"

#include "model/continuation.h"
#include "view/format.h"

#include <algorithm>
#include <format>

namespace game {

namespace {

}

Trainer::Trainer(PetsViewModel& view, const Tr& tr, StatusLine& status, DatasetManager& datasets,
        ModelLibrary& library, ModelSink& models, Parallel parallel)
    : view_(view)
    , tr_(tr)
    , status_(status)
    , datasets_(datasets)
    , library_(library)
    , models_(models)
    , parallel_(std::move(parallel))
    , settings_(view) {
    view_.train.bind_to<Trainer, &Trainer::start, &Trainer::can_start>(*this);
    view_.continueTraining.bind_to<Trainer, &Trainer::continue_training, &Trainer::can_continue>(*this);
    view_.stopTraining.bind_to<Trainer, &Trainer::stop, &Trainer::can_stop>(*this);
    show_dataset();
    show_progress();
    view_.trainBestText = std::string{};
    view_.trainSavedText = std::string{};
}

bool Trainer::settle_name() {
    const std::string typed = view_.modelName.get();
    chosen_name_.clear();
    if (typed.empty()) {
        return true;
    }
    if (status_.reject_name(DatasetStore::check_name(typed))) {
        return false;
    }
    if (library_.contains(typed)) {
        status_.say("status.model_name_taken", {{"name", typed}});
        return false;
    }
    chosen_name_ = typed;
    return true;
}

bool Trainer::can_start() const {
    return !run_.has_value() && datasets_.has_active();
}

bool Trainer::can_continue() const {
    return can_start() && library_.has_selection();
}

std::shared_ptr<const LabeledImages> Trainer::usable_samples() {
    std::shared_ptr<const LabeledImages> samples = datasets_.active_samples();
    if (samples->count(ClassId::First) < kMinPerClass || samples->count(ClassId::Second) < kMinPerClass) {
        status_.say("status.train_no_data", {{"min", std::to_string(kMinPerClass)}});
        return nullptr;
    }
    return samples;
}

void Trainer::start() {
    if (!usable_samples() || !settle_name()) {
        return;
    }
    const TrainingConfig config = settings_.training_config();
    accuracy_before_.reset();
    continued_name_.clear();
    begin(Cnn{settings_.network(), config.seed}, settings_.network(), config, 0, "status.train_started", {});
}

void Trainer::continue_training() {
    const std::shared_ptr<const LabeledImages> samples = usable_samples();
    if (!samples) {
        return;
    }
    if (!settle_name()) {
        return;
    }
    const std::string name = library_.selected_name();
    const auto file = library_.load_selected();
    if (!file) {
        status_.say("status.model_load_failed", {{"detail", file.error()}});
        return;
    }
    TrainingConfig config = settings_.training_config();
    const Continuation plan = plan_continuation(*file, *samples, config.split_seed);
    if (plan.kind == ContinueKind::OtherClasses) {
        status_.say("status.continue_other_classes", {{"model_first", file->class_names[0]},
                {"model_second", file->class_names[1]}, {"first", samples->class_names[0]},
                {"second", samples->class_names[1]}});
        return;
    }
    auto model = Cnn::with_params(file->config, file->params);
    if (!model) {
        status_.say("status.model_load_failed", {{"detail", model.error()}});
        return;
    }

    // The same split as before keeps the check samples out of the training.
    config.split_seed = plan.split_seed;
    accuracy_before_ = file->balanced_accuracy;
    continued_name_ = name;

    const char* key = "status.continue_started";
    if (plan.kind == ContinueKind::UnknownSplit) {
        key = "status.continue_unknown_split";
    } else if (plan.kind == ContinueKind::OtherDataset) {
        key = "status.continue_other_dataset";
    }
    begin(std::move(*model), file->config, config, file->epochs, key, name);
}

void Trainer::begin(Cnn model, const CnnConfig& network, const TrainingConfig& config, unsigned epochs_before,
        const std::string& started_key, const std::string& from_name) {
    const std::shared_ptr<const LabeledImages> samples = datasets_.active_samples();
    running_network_ = network;
    class_names_ = samples->class_names;
    epochs_before_ = epochs_before;
    split_seed_ = config.split_seed;
    dataset_samples_ = samples->size();
    run_.emplace(std::move(model), samples, config, parallel_);
    train_samples_ = run_->train_size();

    shown_.clear();
    rows_.clear();
    view_.epochRows.set(rows_);
    view_.trainBestText = std::string{};
    view_.trainSavedText = std::string{};
    saved_name_.clear();
    set_running(true);
    show_progress();
    status_.say(started_key, {{"name", from_name.empty() ? datasets_.active_name() : from_name},
            {"dataset", datasets_.active_name()}, {"train", std::to_string(run_->train_size())},
            {"val", std::to_string(run_->validation_size())}});
}

void Trainer::stop() {
    run_.reset();
    set_running(false);
    show_progress();
    status_.say("status.train_stopped");
}

void Trainer::set_running(bool running) {
    settings_.set_locked(running);
    view_.training = running;
    view_.notTraining = !running;
}

void Trainer::update() {
    show_dataset();
    if (!run_) {
        return;
    }
    const bool done = run_->step(kSlice);
    show_epochs();
    if (done) {
        finish();
    } else {
        show_progress();
    }
}

void Trainer::finish() {
    const unsigned epochs = run_->completed_epochs();
    const unsigned best_epoch = run_->best_epoch();
    const std::vector<EpochStats> history = run_->history();
    Cnn model = run_->take_model();
    run_.reset();
    set_running(false);
    show_progress();
    if (history.empty()) {
        return;
    }

    const EpochStats& best = history[best_epoch - 1];
    view_.trainBestText = tr_("train.best", {{"epoch", std::to_string(best_epoch)},
            {"loss", std::format("{:.3f}", best.validation_loss)}, {"acc", percent_text(best.balanced_accuracy)}});

    // The epochs of an earlier training count too: the file says how long the network has been trained in all.
    const unsigned total_epochs = epochs_before_ + epochs;
    ModelFile file;
    file.config = running_network_;
    file.epochs = total_epochs;
    file.balanced_accuracy = best.balanced_accuracy;
    file.train_samples = static_cast<unsigned>(train_samples_);
    file.split_known = true;
    file.split_seed = split_seed_;
    file.dataset_samples = static_cast<unsigned>(dataset_samples_);
    file.class_names = class_names_;
    file.params.assign(model.params().begin(), model.params().end());
    const std::string name = chosen_name_.empty() ? ModelStore::new_name(running_network_, total_epochs) : chosen_name_;
    if (const auto saved = library_.save(name, file); !saved) {
        status_.say("status.model_save_failed", {{"detail", saved.error()}});
        return;
    }
    models_.use_model(name, file);
    saved_name_ = name;
    view_.modelName = std::string{};
    view_.trainSavedText = tr_("train.saved", {{"name", name}});
    if (accuracy_before_) {
        status_.say("status.continue_done", {{"from", continued_name_}, {"name", name},
                {"before", percent_text(*accuracy_before_)}, {"after", percent_text(best.balanced_accuracy)}});
    } else {
        status_.say("status.train_done", {{"epoch", std::to_string(best_epoch)},
                {"acc", percent_text(best.balanced_accuracy)}, {"name", name}});
    }
}

void Trainer::apply_locale() {
    dataset_line_written_ = false;
    show_dataset();
    show_progress();
    rows_.clear();
    for (const EpochStats& stats : shown_) {
        auto row = std::make_shared<ListItem>();
        row->label = epoch_text(stats);
        rows_.push_back(std::move(row));
    }
    view_.epochRows.set(rows_);
    if (!saved_name_.empty()) {
        view_.trainSavedText = tr_("train.saved", {{"name", saved_name_}});
    }
}

void Trainer::show_dataset() {
    const std::string name = datasets_.active_name();
    const std::size_t size = datasets_.has_active() ? datasets_.active_samples()->size() : 0;
    if (dataset_line_written_ && name == shown_dataset_ && size == shown_dataset_size_) {
        return;
    }
    dataset_line_written_ = true;
    shown_dataset_ = name;
    shown_dataset_size_ = size;
    view_.trainDatasetText = name.empty() ? tr_("train.dataset_none")
                                          : tr_("train.dataset", {{"name", name}, {"count", std::to_string(size)}});
}

void Trainer::show_progress() {
    if (!run_) {
        view_.trainProgressText = tr_("train.idle");
        return;
    }
    const unsigned epoch = std::min(run_->completed_epochs() + 1, run_->epochs());
    const bool checking = run_->phase() == TrainingRun::Phase::Validating;
    view_.trainProgressText = tr_(checking ? "train.checking" : "train.progress",
            {{"epoch", std::to_string(epoch)}, {"epochs", std::to_string(run_->epochs())},
                    {"percent", percent_text(run_->epoch_progress())}});
}

void Trainer::show_epochs() {
    const std::vector<EpochStats>& history = run_->history();
    if (history.size() == shown_.size()) {
        return;
    }
    for (std::size_t i = shown_.size(); i < history.size(); ++i) {
        auto row = std::make_shared<ListItem>();
        row->label = epoch_text(history[i]);
        rows_.push_back(std::move(row));
    }
    shown_ = history;
    view_.epochRows.set(rows_);
}

std::string Trainer::epoch_text(const EpochStats& stats) const {
    return tr_("train.epoch_row", {{"epoch", std::to_string(stats.epoch)},
            {"loss", std::format("{:.3f}", stats.train_loss)}, {"train", percent_text(stats.train_accuracy)},
            {"val", percent_text(stats.validation_accuracy)}, {"first", class_names_[0]},
            {"first_acc", percent_text(stats.first_accuracy)}, {"second", class_names_[1]},
            {"second_acc", percent_text(stats.second_accuracy)}});
}

}
