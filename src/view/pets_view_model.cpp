#include "view/pets_view_model.h"

namespace game {

PetsViewModel::PetsViewModel() {
    using engine::ui::intern;

    property(intern("statusText"), statusText);
    property(intern("tabRecognize"), tabRecognize);
    property(intern("tabFilters"), tabFilters);
    property(intern("tabTrain"), tabTrain);
    property(intern("tabDatasets"), tabDatasets);
    property(intern("tabHelp"), tabHelp);
    command(intern("showRecognize"), showRecognize);
    command(intern("showFilters"), showFilters);
    command(intern("showTrain"), showTrain);
    command(intern("showDatasets"), showDatasets);
    command(intern("showHelp"), showHelp);
    showRecognize.bind_to<PetsViewModel, &PetsViewModel::show_recognize>(*this);
    showFilters.bind_to<PetsViewModel, &PetsViewModel::show_filters>(*this);
    showTrain.bind_to<PetsViewModel, &PetsViewModel::show_train>(*this);
    showDatasets.bind_to<PetsViewModel, &PetsViewModel::show_datasets>(*this);
    showHelp.bind_to<PetsViewModel, &PetsViewModel::show_help>(*this);
    property(intern("localeIsEn"), localeIsEn);
    property(intern("localeIsUk"), localeIsUk);
    command(intern("localeEn"), localeEn);
    command(intern("localeUk"), localeUk);

    property(intern("sourceText"), sourceText);
    property(intern("outputText"), outputText);
    property(intern("normalizeText"), normalizeText);
    property(intern("normalizing"), normalizing);
    property(intern("notNormalizing"), notNormalizing);
    command(intern("browseSource"), browseSource);
    command(intern("browseOutput"), browseOutput);
    command(intern("normalize"), normalize);
    command(intern("cancelNormalize"), cancelNormalize);

    property(intern("addFileText"), addFileText);
    property(intern("addName"), addName);
    command(intern("browseFile"), browseFile);
    command(intern("addDataset"), addDataset);

    property(intern("datasetTitleText"), datasetTitleText);
    property(intern("datasetCountsText"), datasetCountsText);
    property(intern("pageText"), pageText);
    property(intern("sampleInfoText"), sampleInfoText);
    property(intern("datasets"), datasets);
    property(intern("gridRows"), gridRows);
    command(intern("pagePrev"), pagePrev);
    command(intern("pageNext"), pageNext);
    command(intern("deleteDataset"), deleteDataset);
    paint(intern("samplePaint"), samplePaint);

    property(intern("trainDatasetText"), trainDatasetText);
    property(intern("epochsText"), epochsText);
    property(intern("networkText"), networkText);
    property(intern("augment"), augment);
    property(intern("lrDecay"), lrDecay);
    property(intern("training"), training);
    property(intern("notTraining"), notTraining);
    property(intern("trainProgressText"), trainProgressText);
    property(intern("trainBestText"), trainBestText);
    property(intern("trainSavedText"), trainSavedText);
    property(intern("epochRows"), epochRows);
    command(intern("epochsUp"), epochsUp);
    command(intern("epochsDown"), epochsDown);
    command(intern("networkUp"), networkUp);
    command(intern("networkDown"), networkDown);
    command(intern("train"), train);
    command(intern("continueTraining"), continueTraining);
    command(intern("stopTraining"), stopTraining);

    property(intern("models"), models);
    property(intern("modelInfoText"), modelInfoText);
    property(intern("pictureInfoText"), pictureInfoText);
    property(intern("predictionText"), predictionText);
    command(intern("loadModel"), loadModel);
    command(intern("deleteModel"), deleteModel);
    command(intern("browsePhoto"), browsePhoto);
    command(intern("useSample"), useSample);
    paint(intern("recognizePaint"), recognizePaint);
    property(intern("firstName"), firstName);
    property(intern("secondName"), secondName);
    property(intern("modelName"), modelName);
    paint(intern("firstHeatPaint"), firstHeatPaint);
    paint(intern("secondHeatPaint"), secondHeatPaint);

    property(intern("filtersReady"), filtersReady);
    property(intern("filtersMissing"), filtersMissing);
    property(intern("filterLayerText"), filterLayerText);
    property(intern("filterProgressText"), filterProgressText);
    property(intern("filterNoteText"), filterNoteText);
    property(intern("filterTitleText"), filterTitleText);
    property(intern("filterDetailText"), filterDetailText);
    property(intern("featuresActive"), featuresActive);
    property(intern("featuresIdle"), featuresIdle);
    property(intern("weightsActive"), weightsActive);
    property(intern("weightsIdle"), weightsIdle);
    property(intern("dreamActive"), dreamActive);
    property(intern("dreamIdle"), dreamIdle);
    command(intern("filterBlockUp"), filterBlockUp);
    command(intern("filterBlockDown"), filterBlockDown);
    command(intern("showFeatures"), showFeatures);
    command(intern("showWeights"), showWeights);
    command(intern("showDream"), showDream);
    property(intern("filterRows"), filterRows);
    paint(intern("filterThumbPaint"), filterThumbPaint);
    paint(intern("filterMapPaint"), filterMapPaint);
    paint(intern("filterWeightsPaint"), filterWeightsPaint);
    paint(intern("filterDreamPaint"), filterDreamPaint);
    paint(intern("filterVotesPaint"), filterVotesPaint);
    paint(intern("probabilityPaint"), probabilityPaint);
    paint(intern("idealPaint"), idealPaint);

    property(intern("helpNav"), helpNav);
    property(intern("helpScroll"), helpScroll);
    for (std::size_t i = 0; i < kHelpTopicCount; ++i) {
        property(intern("helpTopic" + std::to_string(i)), helpTopic[i]);
        property(intern("helpShown" + std::to_string(i)), helpShown[i]);
    }
}

}
