#include "recognizer.h"

#include "model/grad_cam.h"
#include "model/picture_file.h"
#include "view/format.h"

#include <format>

namespace game {

Recognizer::Recognizer(PetsViewModel& view, const Tr& tr, StatusLine& status, DatasetManager& datasets,
        ModelLibrary& library, FilterPanel& filters, engine::IWindowControl& windows)
    : view_(view)
    , tr_(tr)
    , status_(status)
    , datasets_(datasets)
    , library_(library)
    , filters_(filters)
    , dialog_(windows)
    , dreamer_(view.idealPaint) {
    view_.loadModel.bind_to<Recognizer, &Recognizer::load_model, &Recognizer::can_load_model>(*this);
    view_.browsePhoto.bind_to<Recognizer, &Recognizer::browse_photo, &Recognizer::can_browse>(*this);
    view_.useSample.bind_to<Recognizer, &Recognizer::use_sample, &Recognizer::can_use_sample>(*this);
    show_texts();
}

void Recognizer::use_model(const std::string& name, const ModelFile& model) {
    auto cnn = Cnn::with_params(model.config, model.params);
    if (!cnn) {
        status_.say("status.model_load_failed", {{"detail", cnn.error()}});
        return;
    }
    model_ = std::make_shared<const Cnn>(std::move(*cnn));
    workspace_ = model_->make_workspace();
    model_name_ = name;
    info_ = model;
    info_.params.clear();
    dreamer_.start(model_);
    filters_.set_model(model_, info_.class_names);
    predict_picture();
}

void Recognizer::load_model() {
    const std::string name = library_.selected_name();
    const auto file = library_.load_selected();
    if (!file) {
        status_.say("status.model_load_failed", {{"detail", file.error()}});
        return;
    }
    use_model(name, *file);
    if (model_ && model_name_ == name) {
        status_.say("status.model_loaded", {{"name", name}});
    }
}

void Recognizer::browse_photo() {
    dialog_.choose_file({engine::FileFilter{tr_("filter.photos"), "jpg;jpeg;png;bmp;gif"},
            engine::FileFilter{tr_("filter.all"), "*"}});
}

void Recognizer::update() {
    dreamer_.collect();
    if (const auto chosen = dialog_.take()) {
        open_picture(*chosen);
    }
}

void Recognizer::open_picture(const std::filesystem::path& path) {
    auto picture = load_normalized_picture(path);
    if (!picture) {
        status_.say("status.picture_failed", {{"detail", picture.error()}});
        return;
    }
    show_picture(std::move(*picture), {"rec.from_file", {{"name", path.filename().string()}}});
}

void Recognizer::use_sample() {
    const auto sample = datasets_.selected_sample();
    const std::string& class_name = datasets_.active_samples()->class_names[static_cast<std::size_t>(sample->species)];
    show_picture({sample->picture.begin(), sample->picture.end()},
            {"rec.from_sample", {{"n", std::to_string(sample->number)}, {"dataset", datasets_.active_name()},
                                        {"class", class_name}}});
}

void Recognizer::show_picture(std::vector<std::uint8_t> picture, Source source) {
    view_.recognizePaint.show(picture);
    view_.firstHeatPaint.show_picture(picture);
    view_.secondHeatPaint.show_picture(picture);
    filters_.set_picture(picture);
    picture_ = std::move(picture);
    source_ = std::move(source);
    predict_picture();
}

void Recognizer::predict_picture() {
    prediction_.reset();
    if (model_ && picture_) {
        prediction_ = predict(*model_, workspace_, *picture_);
        view_.probabilityPaint.show(prediction_->first, prediction_->second);
        const GradCam cam = grad_cam(*model_, workspace_, *picture_);
        view_.firstHeatPaint.show(*picture_, cam.maps[static_cast<std::size_t>(ClassId::First)], cam.side);
        view_.secondHeatPaint.show(*picture_, cam.maps[static_cast<std::size_t>(ClassId::Second)], cam.side);
    } else {
        view_.probabilityPaint.clear();
    }
    show_texts();
}

void Recognizer::show_texts() {
    // Without a network the classes have no names yet.
    const ClassNames names = model_ ? info_.class_names : ClassNames{tr_("class.first"), tr_("class.second")};
    view_.probabilityPaint.set_names(names[0], names[1]);
    view_.idealPaint.set_names(names[0], names[1]);
    view_.firstName = names[0];
    view_.secondName = names[1];

    view_.modelInfoText = model_ ? tr_("rec.model", {{"name", model_name_}, {"network", info_.config.label()},
                                           {"epochs", std::to_string(info_.epochs)},
                                           {"acc", percent_text(info_.balanced_accuracy)}})
                                 : tr_("rec.no_model");
    view_.pictureInfoText = picture_ ? tr_(source_.key, source_.args) : tr_("rec.no_picture");
    if (prediction_) {
        view_.predictionText = tr_("rec.verdict",
                {{"class", names[static_cast<std::size_t>(prediction_->chosen())]},
                        {"percent", percent_text(prediction_->confidence())}});
    } else {
        view_.predictionText = tr_(model_ ? "rec.no_picture" : "rec.no_model_short");
    }
}

}
