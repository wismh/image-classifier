#pragma once

#include "model/cnn.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace game {

// A filter's weights as a small picture to draw. A first-block filter reads colors, so its 3x3 weights are one colored
// picture (what color at each of the nine places makes it fire). A deeper filter reads the maps of the block before:
// one
// 3x3 tile of signed weights per input map, tiles side by side with a one value gap.
struct FilterPicture {
    enum class Kind { Color, Signed };

    Kind kind = Kind::Signed;
    int width = 0;
    int height = 0;
    // Color: width * height * 3 values in 0..1 (0.5 is a weight of zero). Signed: width * height values in -1..1 (0 is
    // a weight of zero, and the gaps). Scaled so that the filter's largest weight, in size, reaches the end of the
    // range.
    std::vector<float> values;
};

[[nodiscard]] FilterPicture filter_weights_picture(const Cnn& model, int block, int filter);

// How strong the weights of a filter are next to the strongest of its block: the length of its weights over the largest
// such length, 0..1.
[[nodiscard]] float filter_strength(const Cnn& model, int block, int filter);

// How much each filter and each hidden neuron pushed the network toward the second class (positive) or the first
// (negative) for one
// picture: its answer times the gradient of (second score - first score) with respect to that answer. Where a layer's
// answers
// reach the scores through ReLUs, poolings and weights without biases, they add up to the whole gap; biases make up the
// rest, so the votes of a layer sum to the gap only roughly.
struct FilterVotes {
    std::array<std::vector<float>, Cnn::kBlocks> conv;  // [block][filter]
    std::vector<float> hidden;                          // [neuron]
    float score_gap = 0.0f;                             // second score - first score
};

// Runs the network on a normalized picture (kImageBytes bytes), leaving every layer's answers in `workspace` (read them
// with activation_map(), hidden_activation() and class_probability()), and works out the votes.
FilterVotes run_filters(const Cnn& model, Cnn::Workspace& workspace, std::span<const std::uint8_t> picture);

// What a filter answered to the picture, after the ReLU: block_side(block) squared values, a view into `workspace`.
[[nodiscard]] std::span<const float> activation_map(const Cnn::Workspace& workspace, int block, int filter);

// The hidden layer. A neuron reads the last block's maps (hidden_inputs() weights, one per map and place); its picture
// is where it looks: for each of the 8 x 8 places, the sum of its weights over all the maps, so red is a place where
// active maps excite it and blue where they hold it back.
[[nodiscard]] FilterPicture hidden_weights_picture(const Cnn& model, int neuron);
[[nodiscard]] float hidden_strength(const Cnn& model, int neuron);
// How the output layer takes the neuron: its weight to the second class's score minus its weight to the first's
// (positive: a
// neuron that votes for the second class whenever it is active). Does not depend on any picture.
[[nodiscard]] float hidden_connection(const Cnn& model, int neuron);
// What a hidden neuron answered, after its ReLU.
[[nodiscard]] float hidden_activation(const Cnn::Workspace& workspace, int neuron);

// The output layer: the hidden neurons' weights to one species' score, laid out in a grid as near to a square as it
// comes.
[[nodiscard]] FilterPicture output_weights_picture(const Cnn& model, ClassId species);
// The probability of a species in the last run (the softmax of the scores).
[[nodiscard]] float class_probability(const Cnn::Workspace& workspace, ClassId species);

// The mean of a map.
[[nodiscard]] float mean_activation(std::span<const float> map);

}
