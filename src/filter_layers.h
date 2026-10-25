#pragma once

#include "model/cnn.h"

namespace game {

// The layers of the network that have units to look at: the convolutions of the blocks, then the hidden layer, then the
// output layer (one unit for each species).
struct FilterLayers {
    static constexpr int kCount = Cnn::kBlocks + 2;
    static constexpr int kHidden = Cnn::kBlocks;
    static constexpr int kOutput = Cnn::kBlocks + 1;

    [[nodiscard]] static bool is_conv(int layer) { return layer < Cnn::kBlocks; }
    [[nodiscard]] static int units(const Cnn& model, int layer) {
        return is_conv(layer) ? model.filters(layer) : layer == kHidden ? model.hidden_neurons()
                                                                         : static_cast<int>(kNumClasses);
    }
};

}
