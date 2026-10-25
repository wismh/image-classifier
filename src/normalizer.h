#pragma once

#include "dataset_manager.h"
#include "loc/localized.h"
#include "loc/tr.h"
#include "model/normalization_run.h"
#include "path_dialog.h"
#include "status_line.h"
#include "view/pets_view_model.h"

#include <filesystem>
#include <optional>

namespace game {

// Step 1 of the Datasets tab: turns the photos of a folder into one normalized dataset file. The folder and the place
// to
// save to are picked in native dialogs. The work is spread over the frames (update() once per frame), so the window
// stays responsive and the run can be cancelled. The result is handed to step 2.
class Normalizer final : public Localized {
public:
    // Photos handled per frame: enough to keep the worker threads busy, few enough to keep the frame short.
    static constexpr std::size_t kPhotosPerFrame = 24;

    Normalizer(PetsViewModel& view, const Tr& tr, StatusLine& status, DatasetManager& datasets,
            engine::IWindowControl& windows, Parallel parallel);

    Normalizer(const Normalizer&) = delete;
    Normalizer& operator=(const Normalizer&) = delete;

    // Writes the texts under the buttons again after a change of language.
    void apply_locale() override;
    void update();

private:
    enum class State { Idle, Running, Done, Stopped };
    enum class Target { Source, Output };

    void browse_source();
    void browse_output();
    [[nodiscard]] bool can_browse() const { return !dialog_.pending() && !run_.has_value(); }
    void start();
    void cancel();
    [[nodiscard]] bool can_start() const { return !run_.has_value(); }
    [[nodiscard]] bool can_cancel() const { return run_.has_value(); }
    void finish();
    void show_state();
    void show_paths();

    PetsViewModel& view_;
    const Tr& tr_;
    StatusLine& status_;
    DatasetManager& datasets_;
    Parallel parallel_;
    PathDialog dialog_;
    Target target_ = Target::Source;
    std::filesystem::path source_;
    std::filesystem::path output_directory_;
    std::optional<NormalizationRun> run_;
    std::filesystem::path output_;
    State state_ = State::Idle;
    std::size_t kept_ = 0;
    std::size_t skipped_ = 0;
};

}
