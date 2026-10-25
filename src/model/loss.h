#pragma once

#include <span>

namespace game {

struct LossResult {
    float loss = 0.0f;       // -log of the probability the network gave the right class
    int predicted = 0;       // the class with the highest score
};

// Softmax over `scores`, then cross-entropy against the class `label`. Writes the gradient of the loss with respect
// to the scores, multiplied by `weight`, into `d_scores` (softmax minus one-hot, times weight). The loss itself is
// returned unweighted.
LossResult softmax_cross_entropy(std::span<const float> scores, int label, float weight, std::span<float> d_scores);

}
