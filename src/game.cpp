#include "game.h"

#include <asset_ids.h>

#include "loc/locale_prefs.h"
#include "model/model_store.h"

#include <engine/core/platform.h>
#include <engine/loc/catalog.h>
#include <engine/log.h>

namespace game {

namespace {

constexpr glm::vec2 kDesignSize{920.0f, 640.0f};

// The domain runs its loops on the engine's worker threads through this.
Parallel parallel_over(engine::JobSystem& jobs) {
    return Parallel{static_cast<std::size_t>(jobs.worker_count()) + 1,
            [&jobs](std::size_t count, std::size_t grain, const Parallel::RangeFn& fn) {
                jobs.parallel_for(count, grain, [&fn](std::size_t begin, std::size_t end) { fn(begin, end); });
            }};
}

}

Game::Game(const engine::EngineServices& services)
    : engine::GameBase(services.worlds)
    , jobs_(&services.jobs)
    , windows_(&services.windows)
    , assets_(&services.assets) {}

engine::WindowDesc Game::primary_window() const {
    return engine::WindowDesc{.title = "Image Classifier",
            .size = {static_cast<int>(kDesignSize.x), static_cast<int>(kDesignSize.y)},
            .style = {.resizable = false}};
}

std::optional<engine::AssetId> Game::window_icon() const {
    return assets::textures::app_icon;
}

void Game::on_start() {
    // Normalized datasets live in <user data>/datasets, trained networks in <user data>/models. Without a user data
    // directory nothing can be added or saved.
    std::filesystem::path dataset_directory;
    std::filesystem::path model_directory;
    if (const auto directory = engine::user_data_directory("wismh", "cats-vs-dogs")) {
        dataset_directory = *directory / "datasets";
        model_directory = *directory / "models";
        preferences_directory_ = *directory;
    } else {
        engine::log::warn("no user data directory: datasets cannot be added");
    }

    load_locale();
    view_ = std::make_shared<PetsViewModel>();
    const auto& catalog = world().ctx<engine::loc::Catalog>();
    view_->localeIsEn = catalog.active() == "en";
    view_->localeIsUk = catalog.active() == "uk";
    view_->localeEn.bind_to<Game, &Game::use_english>(*this);
    view_->localeUk.bind_to<Game, &Game::use_ukrainian>(*this);

    status_.emplace(*view_, *tr_);
    datasets_.emplace(*view_, DatasetStore{std::move(dataset_directory)}, *tr_, *status_, *windows_);
    normalizer_.emplace(*view_, *tr_, *status_, *datasets_, *windows_, parallel_over(*jobs_));
    library_.emplace(*view_, ModelStore{std::move(model_directory)}, *tr_, *status_);
    filters_.emplace(*view_, *tr_);
    recognizer_.emplace(*view_, *tr_, *status_, *datasets_, *library_, *filters_, *windows_);
    trainer_.emplace(*view_, *tr_, *status_, *datasets_, *library_, *recognizer_, parallel_over(*jobs_));
    help_.emplace(*view_, *tr_);
    localized_ = {&*normalizer_, &*datasets_, &*trainer_, &*recognizer_, &*filters_, &*help_};
    status_->say("status.ready");

    engine::ui::UiCanvas canvas;
    canvas.document = assets::ui::pets;
    canvas.stylesheet = assets::css::pets;
    canvas.fit = engine::ui::UiFit::ScaleWithScreenSize;
    canvas.reference_size = kDesignSize;
    canvas.data_context = view_;
    screen_ = world().create();
    world().emplace<engine::ui::UiCanvas>(screen_, std::move(canvas));

    world().add_system(engine::ecs::Schedule::Frame, engine::ecs::Phase::Game,
            [this](engine::ecs::World&) {
                normalizer_->update();
                datasets_->update();
                recognizer_->update();
                filters_->update(view_->tabFilters.get());
                trainer_->update();
            });
}

void Game::load_locale() {
    auto& catalog = world().ctx<engine::loc::Catalog>();
    catalog.add(*assets_->get<engine::loc::StringTable>(assets::locale::en), engine::loc::Role::Source);
    catalog.add(*assets_->get<engine::loc::StringTable>(assets::locale::uk));
    catalog.set_fallback("en");
    catalog.set_active(read_locale(preferences_directory_, "en"));
    tr_.emplace(catalog);
}

void Game::set_locale(std::string_view tag) {
    auto& catalog = world().ctx<engine::loc::Catalog>();
    if (catalog.active() == tag) {
        return;
    }
    catalog.set_active(std::string{tag});
    write_locale(preferences_directory_, tag);
    view_->localeIsEn = tag == "en";
    view_->localeIsUk = tag == "uk";
    for (Localized* panel : localized_) {
        panel->apply_locale();
    }
    status_->refresh();
}

}
