#include "model/grad_cam.h"

#include "model/augment.h"

#include <algorithm>

namespace game {

GradCam grad_cam(const Cnn& model, Cnn::Workspace& workspace, std::span<const std::uint8_t> picture) {
    std::vector<float> input(Cnn::kInputSize);
    to_input(picture, input);
    (void)model.forward(input, workspace);

    constexpr int kLast = Cnn::kBlocks - 1;
    const std::size_t maps = static_cast<std::size_t>(model.filters(kLast));
    const std::size_t plane = static_cast<std::size_t>(Cnn::block_side(kLast)) * Cnn::block_side(kLast);
    const std::size_t layer = Cnn::relu_layer(kLast);

    GradCam cam;
    cam.side = Cnn::block_side(kLast);
    std::vector<float> unused(model.param_count());
    float strongest = 0.0f;
    for (std::size_t species = 0; species < kNumClasses; ++species) {
        // The gradient of this species' lead: its score minus the other one's.
        std::array<float, kNumClasses> d_scores{};
        d_scores.fill(-1.0f);
        d_scores[species] = 1.0f;
        std::ranges::fill(unused, 0.0f);
        model.backward(input, workspace, d_scores, unused);

        // The gradient with respect to a layer's output is the gradient with respect to the next layer's input.
        const std::vector<float>& answers = workspace.outputs[layer];
        const std::vector<float>& gradients = workspace.deltas[layer + 1];
        std::vector<float>& map = cam.maps[species];
        map.assign(plane, 0.0f);
        for (std::size_t filter = 0; filter < maps; ++filter) {
            float weight = 0.0f;
            for (std::size_t i = filter * plane; i < (filter + 1) * plane; ++i) {
                weight += gradients[i];
            }
            weight /= static_cast<float>(plane);
            for (std::size_t place = 0; place < plane; ++place) {
                map[place] += weight * answers[filter * plane + place];
            }
        }
        for (float& value : map) {
            value = std::max(value, 0.0f);
            strongest = std::max(strongest, value);
        }
    }
    if (strongest > 0.0f) {
        for (auto& map : cam.maps) {
            for (float& value : map) {
                value /= strongest;
            }
        }
    }
    return cam;
}

}
