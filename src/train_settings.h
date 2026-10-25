#pragma once

#include "model/cnn.h"
#include "model/training_run.h"
#include "view/pets_view_model.h"

#include <array>

namespace game {

// The steppers and the checkbox of the Train tab: how many epochs, how big the network, and whether the pictures are
// changed at random while training. Locked while a training runs.
class TrainSettings {
public:
    static constexpr unsigned kMinEpochs = 1;
    static constexpr unsigned kMaxEpochs = 50;
    // The networks to pick from, smallest to largest.
    static constexpr std::array<CnnConfig, 3> kNetworks{
            CnnConfig{4, 8, 16, 32}, CnnConfig{8, 16, 32, 64}, CnnConfig{16, 32, 64, 128}};

    explicit TrainSettings(PetsViewModel& view);

    TrainSettings(const TrainSettings&) = delete;
    TrainSettings& operator=(const TrainSettings&) = delete;

    void set_locked(bool locked) { locked_ = locked; }

    [[nodiscard]] unsigned epochs() const { return epochs_; }
    [[nodiscard]] const CnnConfig& network() const { return kNetworks[network_]; }
    [[nodiscard]] bool augment() const { return view_.augment.get(); }
    [[nodiscard]] bool lr_decay() const { return view_.lrDecay.get(); }
    // The settings of a run with fresh seeds for the order of samples and the split of the dataset.
    [[nodiscard]] TrainingConfig training_config() const;

private:
    [[nodiscard]] bool can_epochs_up() const { return !locked_ && epochs_ < kMaxEpochs; }
    [[nodiscard]] bool can_epochs_down() const { return !locked_ && epochs_ > kMinEpochs; }
    [[nodiscard]] bool can_network_up() const { return !locked_ && network_ + 1 < kNetworks.size(); }
    [[nodiscard]] bool can_network_down() const { return !locked_ && network_ > 0; }

    void epochs_up();
    void epochs_down();
    void network_up();
    void network_down();
    void show();

    PetsViewModel& view_;
    bool locked_ = false;
    unsigned epochs_ = 10;
    std::size_t network_ = 1;
};

}
