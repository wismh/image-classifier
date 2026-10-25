#pragma once

#include "help/help_topic_table.h"
#include "help/help_view_model.h"
#include "view/grid_row.h"
#include "view/heat_paint.h"
#include "view/filter_paint.h"
#include "view/filter_row.h"
#include "view/ideal_paint.h"
#include "view/list_item.h"
#include "view/probability_paint.h"
#include "view/votes_paint.h"
#include "view/sample_paint.h"

#include <engine/ui/view_model.h>

#include <array>
#include <memory>
#include <string>

namespace game {

// Everything pets.xml binds. Behavior lives in Normalizer and DatasetManager; this type only holds state and the
// paints.
class PetsViewModel final : public engine::ui::ViewModel {
public:
    PetsViewModel();

    // Window-wide.
    engine::ui::Bindable<std::string> statusText;
    engine::ui::Bindable<bool> tabRecognize{true};
    engine::ui::Bindable<bool> tabFilters{false};
    engine::ui::Bindable<bool> tabTrain{false};
    engine::ui::Bindable<bool> tabDatasets{false};
    engine::ui::Bindable<bool> tabHelp{false};
    engine::ui::MethodCommand showRecognize;
    engine::ui::MethodCommand showFilters;
    engine::ui::MethodCommand showTrain;
    engine::ui::MethodCommand showDatasets;
    engine::ui::MethodCommand showHelp;
    // Language buttons: the commands are bound by the Game, which owns the catalog.
    engine::ui::Bindable<bool> localeIsEn{true};
    engine::ui::Bindable<bool> localeIsUk{false};
    engine::ui::MethodCommand localeEn;
    engine::ui::MethodCommand localeUk;

    // Step 1 of the Datasets tab: photos to a normalized file.
    engine::ui::Bindable<std::string> sourceText;
    engine::ui::Bindable<std::string> outputText;
    engine::ui::Bindable<std::string> normalizeText;
    engine::ui::Bindable<bool> normalizing{false};
    engine::ui::Bindable<bool> notNormalizing{true};
    engine::ui::MethodCommand browseSource;
    engine::ui::MethodCommand browseOutput;
    engine::ui::MethodCommand normalize;
    engine::ui::MethodCommand cancelNormalize;

    // Step 2: a normalized file to a dataset of the app.
    engine::ui::Bindable<std::string> addFileText;
    engine::ui::Bindable<std::string> addName;
    engine::ui::MethodCommand browseFile;
    engine::ui::MethodCommand addDataset;

    // The datasets of the app.
    engine::ui::Bindable<std::string> datasetTitleText;
    engine::ui::Bindable<std::string> datasetCountsText;
    engine::ui::Bindable<std::string> pageText;
    engine::ui::Bindable<std::string> sampleInfoText;
    engine::ui::BindableList<std::shared_ptr<ListItem>> datasets;
    engine::ui::BindableList<std::shared_ptr<GridRow>> gridRows;
    engine::ui::MethodCommand pagePrev;
    engine::ui::MethodCommand pageNext;
    engine::ui::MethodCommand deleteDataset;
    SamplePaint samplePaint;

    // The Train tab.
    engine::ui::Bindable<std::string> trainDatasetText;
    engine::ui::Bindable<std::string> epochsText{"10"};
    engine::ui::Bindable<std::string> networkText;
    engine::ui::Bindable<bool> augment{true};
    engine::ui::Bindable<bool> lrDecay{false};
    // What the network about to be trained is saved as; empty: a name is made up.
    engine::ui::Bindable<std::string> modelName;
    engine::ui::Bindable<bool> training{false};
    engine::ui::Bindable<bool> notTraining{true};
    engine::ui::Bindable<std::string> trainProgressText;
    engine::ui::Bindable<std::string> trainBestText;
    engine::ui::Bindable<std::string> trainSavedText;
    engine::ui::BindableList<std::shared_ptr<ListItem>> epochRows;
    engine::ui::MethodCommand epochsUp;
    engine::ui::MethodCommand epochsDown;
    engine::ui::MethodCommand networkUp;
    engine::ui::MethodCommand networkDown;
    engine::ui::MethodCommand train;
    engine::ui::MethodCommand continueTraining;
    engine::ui::MethodCommand stopTraining;

    // The Recognize tab.
    engine::ui::BindableList<std::shared_ptr<ListItem>> models;
    engine::ui::Bindable<std::string> modelInfoText;
    engine::ui::Bindable<std::string> pictureInfoText;
    engine::ui::Bindable<std::string> predictionText;
    engine::ui::MethodCommand loadModel;
    engine::ui::MethodCommand deleteModel;
    engine::ui::MethodCommand browsePhoto;
    engine::ui::MethodCommand useSample;
    SamplePaint recognizePaint;
    // What the two classes of the network on show are called.
    engine::ui::Bindable<std::string> firstName;
    engine::ui::Bindable<std::string> secondName;
    HeatPaint firstHeatPaint;
    HeatPaint secondHeatPaint;

    // The Filters tab. The three kinds of picture each have a button that is plain when the kind is not on show and
    // marked when it is.
    engine::ui::Bindable<bool> filtersReady{false};
    engine::ui::Bindable<bool> filtersMissing{true};
    engine::ui::Bindable<std::string> filterLayerText;
    engine::ui::Bindable<std::string> filterProgressText;
    engine::ui::Bindable<std::string> filterNoteText;
    engine::ui::Bindable<std::string> filterTitleText;
    engine::ui::Bindable<std::string> filterDetailText;
    engine::ui::Bindable<bool> featuresActive{false};
    engine::ui::Bindable<bool> featuresIdle{true};
    engine::ui::Bindable<bool> weightsActive{false};
    engine::ui::Bindable<bool> weightsIdle{true};
    engine::ui::Bindable<bool> dreamActive{false};
    engine::ui::Bindable<bool> dreamIdle{true};
    engine::ui::MethodCommand filterBlockUp;
    engine::ui::MethodCommand filterBlockDown;
    engine::ui::MethodCommand showFeatures;
    engine::ui::MethodCommand showWeights;
    engine::ui::MethodCommand showDream;
    engine::ui::BindableList<std::shared_ptr<FilterRow>> filterRows;
    SamplePaint filterThumbPaint;
    FilterPaint filterMapPaint;
    FilterPaint filterWeightsPaint;
    FilterPaint filterDreamPaint;
    VotesPaint filterVotesPaint;
    ProbabilityPaint probabilityPaint;
    IdealPaint idealPaint;

    // The Help tab. Topic i is a one-item list so that its document is built from the template; helpShown[i] says
    // whether
    // it is the one on show.
    engine::ui::BindableList<std::shared_ptr<HelpNavItem>> helpNav;
    engine::ui::Bindable<float> helpScroll{0.0f};
    std::array<engine::ui::BindableList<std::shared_ptr<HelpTopicViewModel>>, kHelpTopicCount> helpTopic;
    std::array<engine::ui::Bindable<bool>, kHelpTopicCount> helpShown;

private:
    enum class Tab { Recognize, Filters, Train, Datasets, Help };

    void show_tab(Tab tab) {
        tabRecognize = tab == Tab::Recognize;
        tabFilters = tab == Tab::Filters;
        tabTrain = tab == Tab::Train;
        tabDatasets = tab == Tab::Datasets;
        tabHelp = tab == Tab::Help;
    }
    void show_recognize() { show_tab(Tab::Recognize); }
    void show_filters() { show_tab(Tab::Filters); }
    void show_train() { show_tab(Tab::Train); }
    void show_datasets() { show_tab(Tab::Datasets); }
    void show_help() { show_tab(Tab::Help); }
};

}
