#include "model/prediction.h"

#include "model/augment.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace game {

Prediction predict(const Cnn& model, Cnn::Workspace& workspace, std::span<const std::uint8_t> picture) {
    std::vector<float> input(Cnn::kInputSize);
    to_input(picture, input);
    const std::span<const float> scores = model.forward(input, workspace);

    // Softmax; subtracting the top score keeps exp() from overflowing.
    const float top = std::max(scores[0], scores[1]);
    const float first = std::exp(scores[static_cast<std::size_t>(ClassId::First)] - top);
    const float second = std::exp(scores[static_cast<std::size_t>(ClassId::Second)] - top);
    return {first / (first + second), second / (first + second)};
}

}
