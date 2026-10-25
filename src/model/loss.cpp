#include "model/loss.h"

#include <algorithm>
#include <cmath>

namespace game {

LossResult softmax_cross_entropy(std::span<const float> scores, int label, float weight, std::span<float> d_scores) {
    const auto highest = std::ranges::max_element(scores);
    const float top = *highest;
    // Subtracting the top score keeps exp() from overflowing.
    float sum = 0.0f;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        d_scores[i] = std::exp(scores[i] - top);
        sum += d_scores[i];
    }
    for (std::size_t i = 0; i < scores.size(); ++i) {
        d_scores[i] /= sum;
    }
    // log(p) = (score - top) - log(sum); computed from the scores so a tiny p does not become -inf.
    const float loss = std::log(sum) - (scores[static_cast<std::size_t>(label)] - top);
    d_scores[static_cast<std::size_t>(label)] -= 1.0f;
    for (float& value : d_scores) {
        value *= weight;
    }
    return {loss, static_cast<int>(highest - scores.begin())};
}

}
