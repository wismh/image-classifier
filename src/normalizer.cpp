#include "normalizer.h"

#include "model/dataset_file.h"
#include "model/output_path.h"
#include "model/path_display.h"

namespace game {

Normalizer::Normalizer(PetsViewModel& view, const Tr& tr, StatusLine& status, DatasetManager& datasets,
        engine::IWindowControl& windows, Parallel parallel)
    : view_(view)
    , tr_(tr)
    , status_(status)
    , datasets_(datasets)
    , parallel_(std::move(parallel))
    , dialog_(windows) {
    view_.browseSource.bind_to<Normalizer, &Normalizer::browse_source, &Normalizer::can_browse>(*this);
    view_.browseOutput.bind_to<Normalizer, &Normalizer::browse_output, &Normalizer::can_browse>(*this);
    view_.normalize.bind_to<Normalizer, &Normalizer::start, &Normalizer::can_start>(*this);
    view_.cancelNormalize.bind_to<Normalizer, &Normalizer::cancel, &Normalizer::can_cancel>(*this);
    show_paths();
    show_state();
}

void Normalizer::apply_locale() {
    show_paths();
    show_state();
}

void Normalizer::browse_source() {
    target_ = Target::Source;
    dialog_.choose_folder(source_.empty() ? std::filesystem::path{} : source_.parent_path());
}

void Normalizer::browse_output() {
    target_ = Target::Output;
    dialog_.choose_folder(output_directory_);
}

void Normalizer::show_paths() {
    view_.sourceText = source_.empty() ? tr_("dialog.none") : display_path(source_, DatasetManager::kShownPathLength);
    const std::filesystem::path planned = planned_output_path(source_, output_directory_);
    view_.outputText = planned.empty() ? tr_("dialog.output_default")
                                       : display_path(planned, DatasetManager::kShownPathLength);
}

void Normalizer::start() {
    if (source_.empty()) {
        status_.say("status.folder_missing");
        return;
    }
    auto scan = scan_source_folder(source_);
    if (!scan) {
        status_.say("status.norm_failed", {{"detail", scan.error()}});
        return;
    }

    output_ = planned_output_path(source_, output_directory_);
    status_.say("status.norm_started", {{"count", std::to_string(scan->files.size())}});
    run_.emplace(std::move(*scan), parallel_);
    state_ = State::Running;
    view_.normalizing = true;
    view_.notNormalizing = false;
    show_state();
}

void Normalizer::cancel() {
    run_.reset();
    state_ = State::Stopped;
    view_.normalizing = false;
    view_.notNormalizing = true;
    show_state();
    status_.say("status.norm_cancelled");
}

void Normalizer::update() {
    if (const auto chosen = dialog_.take()) {
        (target_ == Target::Source ? source_ : output_directory_) = *chosen;
        show_paths();
    }
    if (!run_) {
        return;
    }
    run_->step(kPhotosPerFrame);
    if (run_->done()) {
        finish();
    } else {
        show_state();
    }
}

void Normalizer::finish() {
    skipped_ = run_->skipped();
    LabeledImages data = run_->take_result();
    run_.reset();
    view_.normalizing = false;
    view_.notNormalizing = true;
    kept_ = data.size();

    if (data.size() == 0) {
        state_ = State::Stopped;
        show_state();
        status_.say("status.norm_nothing");
        return;
    }
    if (const auto written = write_dataset_file(output_, data); !written) {
        state_ = State::Stopped;
        show_state();
        status_.say("status.norm_write_failed", {{"detail", written.error()}});
        return;
    }
    state_ = State::Done;
    show_state();
    status_.say("status.norm_saved", {{"count", std::to_string(kept_)}, {"path", output_.string()}});

    // Step 2 starts from this file.
    datasets_.offer_file(output_);
}

void Normalizer::show_state() {
    switch (state_) {
    case State::Idle: view_.normalizeText = tr_("norm.idle"); break;
    case State::Running:
        view_.normalizeText = tr_("norm.progress",
                {{"done", std::to_string(run_->processed())}, {"total", std::to_string(run_->total())},
                        {"skipped", std::to_string(run_->skipped())}});
        break;
    case State::Done:
        view_.normalizeText =
                tr_("norm.done", {{"count", std::to_string(kept_)}, {"skipped", std::to_string(skipped_)}});
        break;
    case State::Stopped: view_.normalizeText = tr_("norm.failed"); break;
    }
}

}
