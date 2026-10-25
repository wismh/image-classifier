#include "train_settings.h"

#include <random>
#include <string>

namespace game {

TrainSettings::TrainSettings(PetsViewModel& view)
    : view_(view) {
    view_.epochsUp.bind_to<TrainSettings, &TrainSettings::epochs_up, &TrainSettings::can_epochs_up>(*this);
    view_.epochsDown.bind_to<TrainSettings, &TrainSettings::epochs_down, &TrainSettings::can_epochs_down>(*this);
    view_.networkUp.bind_to<TrainSettings, &TrainSettings::network_up, &TrainSettings::can_network_up>(*this);
    view_.networkDown.bind_to<TrainSettings, &TrainSettings::network_down, &TrainSettings::can_network_down>(*this);
    show();
}

TrainingConfig TrainSettings::training_config() const {
    TrainingConfig config;
    config.epochs = epochs_;
    config.augment = augment();
    config.lr_decay = lr_decay();
    config.seed = std::random_device{}();
    config.split_seed = std::random_device{}();
    return config;
}

void TrainSettings::epochs_up() {
    ++epochs_;
    show();
}

void TrainSettings::epochs_down() {
    --epochs_;
    show();
}

void TrainSettings::network_up() {
    ++network_;
    show();
}

void TrainSettings::network_down() {
    --network_;
    show();
}

void TrainSettings::show() {
    view_.epochsText = std::to_string(epochs_);
    view_.networkText = network().label();
}

}
