#pragma once

#include "model/cnn.h"

#include <cstdint>
#include <span>

namespace game {

// What the network thinks of a picture: the probability of each class, adding up to 1.
struct Prediction {
    float first = 0.5f;
    float second = 0.5f;

    [[nodiscard]] ClassId chosen() const { return second > first ? ClassId::Second : ClassId::First; }
    // The probability of the class the network chose.
    [[nodiscard]] float confidence() const { return second > first ? second : first; }
};

// Runs `model` on a normalized picture (kImageBytes bytes, as in a dataset). `workspace` must come from
// model.make_workspace(); it is the caller's so that calls from several threads do not share one.
[[nodiscard]] Prediction predict(const Cnn& model, Cnn::Workspace& workspace, std::span<const std::uint8_t> picture);

}
