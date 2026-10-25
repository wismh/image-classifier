#pragma once

#include "dataset_manager.h"
#include "filter_panel.h"
#include "help/help.h"
#include "loc/tr.h"
#include "model_library.h"
#include "normalizer.h"
#include "recognizer.h"
#include "status_line.h"
#include "trainer.h"
#include "view/pets_view_model.h"

#include <engine/core/engine_services.h>
#include <engine/core/job_system.h>
#include <engine/resources/assets_db.h>
#include <engine/igame.h>
#include <engine/ui/canvas.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace game {

// One window with the pets screen. Behavior lives in Normalizer (photos to a normalized file) and DatasetManager
// (normalized files to the datasets of the app); the game owns the lifecycle and the language.
class Game final : public engine::GameBase {
public:
    explicit Game(const engine::EngineServices& services);

    [[nodiscard]] engine::WindowDesc primary_window() const override;
    [[nodiscard]] std::optional<engine::AssetId> window_icon() const override;
    void on_start() override;

private:
    // Loads the English and Ukrainian messages and picks the language the player used last time.
    void load_locale();
    void set_locale(std::string_view tag);
    void use_english() { set_locale("en"); }
    void use_ukrainian() { set_locale("uk"); }

    engine::JobSystem* jobs_;
    engine::IWindowControl* windows_;
    engine::AssetsDb* assets_;
    std::filesystem::path preferences_directory_;
    std::optional<Tr> tr_;
    std::shared_ptr<PetsViewModel> view_;
    std::optional<StatusLine> status_;
    std::optional<Normalizer> normalizer_;
    std::optional<DatasetManager> datasets_;
    std::optional<ModelLibrary> library_;
    std::optional<FilterPanel> filters_;
    std::optional<Recognizer> recognizer_;
    std::optional<Trainer> trainer_;
    std::optional<Help> help_;
    std::vector<Localized*> localized_;  // every panel that has texts of its own
    engine::ecs::Entity screen_{};
};

}
