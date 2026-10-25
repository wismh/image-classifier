#include "model/filter_view.h"

#include "model/augment.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace game {

namespace {

// The 3x3 weights of one (filter, input) pair.
std::span<const float> kernel(const Cnn& model, int block, int filter, int input) {
    const std::size_t inputs = static_cast<std::size_t>(model.block_inputs(block));
    return model.conv_weights(block).subspan((static_cast<std::size_t>(filter) * inputs + input) * 9, 9);
}

std::span<const float> filter_weights(const Cnn& model, int block, int filter) {
    const std::size_t inputs = static_cast<std::size_t>(model.block_inputs(block));
    return model.conv_weights(block).subspan(static_cast<std::size_t>(filter) * inputs * 9, inputs * 9);
}

float largest_magnitude(std::span<const float> values) {
    float largest = 0.0f;
    for (const float value : values) {
        largest = std::max(largest, std::abs(value));
    }
    return largest;
}

float length(std::span<const float> values) {
    double sum = 0.0;
    for (const float value : values) {
        sum += static_cast<double>(value) * value;
    }
    return static_cast<float>(std::sqrt(sum));
}

}

FilterPicture filter_weights_picture(const Cnn& model, int block, int filter) {
    FilterPicture picture;
    const float scale = largest_magnitude(filter_weights(model, block, filter));
    const float unit = scale > 0.0f ? 1.0f / scale : 0.0f;

    if (block == 0) {
        picture.kind = FilterPicture::Kind::Color;
        picture.width = 3;
        picture.height = 3;
        picture.values.assign(9 * 3, 0.5f);
        for (int c = 0; c < kChannels; ++c) {
            const std::span<const float> weights = kernel(model, 0, filter, c);
            for (int i = 0; i < 9; ++i) {
                picture.values[static_cast<std::size_t>(i) * 3 + c] = 0.5f + 0.5f * weights[static_cast<std::size_t>(i)] * unit;
            }
        }
        return picture;
    }

    const int inputs = model.block_inputs(block);
    const int columns = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(inputs))));
    const int rows = (inputs + columns - 1) / columns;
    picture.kind = FilterPicture::Kind::Signed;
    picture.width = columns * 3 + (columns - 1);
    picture.height = rows * 3 + (rows - 1);
    picture.values.assign(static_cast<std::size_t>(picture.width) * picture.height, 0.0f);
    for (int input = 0; input < inputs; ++input) {
        const std::span<const float> weights = kernel(model, block, filter, input);
        const int left = (input % columns) * 4;
        const int top = (input / columns) * 4;
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 3; ++x) {
                picture.values[static_cast<std::size_t>(top + y) * picture.width + left + x] =
                        weights[static_cast<std::size_t>(y * 3 + x)] * unit;
            }
        }
    }
    return picture;
}

float filter_strength(const Cnn& model, int block, int filter) {
    float strongest = 0.0f;
    for (int other = 0; other < model.filters(block); ++other) {
        strongest = std::max(strongest, length(filter_weights(model, block, other)));
    }
    return strongest > 0.0f ? length(filter_weights(model, block, filter)) / strongest : 0.0f;
}

namespace {

// The sum over a layer's units of answer * gradient, for each unit: `plane` values each.
std::vector<float> votes_of(std::span<const float> answers, std::span<const float> gradients, std::size_t plane) {
    std::vector<float> votes(answers.size() / plane, 0.0f);
    for (std::size_t unit = 0; unit < votes.size(); ++unit) {
        for (std::size_t i = unit * plane; i < (unit + 1) * plane; ++i) {
            votes[unit] += answers[i] * gradients[i];
        }
    }
    return votes;
}

}

FilterVotes run_filters(const Cnn& model, Cnn::Workspace& workspace, std::span<const std::uint8_t> picture) {
    std::vector<float> input(Cnn::kInputSize);
    to_input(picture, input);
    const std::span<const float> scores = model.forward(input, workspace);

    // The gradient of (second score - first score) with respect to every layer's answers.
    std::vector<float> d_scores(kNumClasses, 0.0f);
    d_scores[static_cast<std::size_t>(ClassId::First)] = -1.0f;
    d_scores[static_cast<std::size_t>(ClassId::Second)] = 1.0f;
    std::vector<float> unused(model.param_count(), 0.0f);
    model.backward(input, workspace, d_scores, unused);

    FilterVotes votes;
    votes.score_gap = scores[static_cast<std::size_t>(ClassId::Second)] - scores[static_cast<std::size_t>(ClassId::First)];
    for (int block = 0; block < Cnn::kBlocks; ++block) {
        const std::size_t layer = Cnn::relu_layer(block);
        const std::size_t plane = static_cast<std::size_t>(Cnn::block_side(block)) * Cnn::block_side(block);
        // The gradient with respect to a layer's output is the gradient with respect to the next layer's input.
        votes.conv[static_cast<std::size_t>(block)] = votes_of(workspace.outputs[layer], workspace.deltas[layer + 1], plane);
    }
    votes.hidden = votes_of(workspace.outputs[Cnn::hidden_relu_layer()], workspace.deltas[Cnn::output_layer()], 1);
    return votes;
}

FilterPicture hidden_weights_picture(const Cnn& model, int neuron) {
    const std::size_t inputs = model.hidden_inputs();
    const int maps = model.filters(Cnn::kBlocks - 1);
    const std::size_t plane = inputs / static_cast<std::size_t>(maps);
    const int side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(plane))));
    const std::span<const float> weights = model.hidden_weights().subspan(static_cast<std::size_t>(neuron) * inputs, inputs);

    FilterPicture picture;
    picture.kind = FilterPicture::Kind::Signed;
    picture.width = side;
    picture.height = side;
    picture.values.assign(plane, 0.0f);
    for (int map = 0; map < maps; ++map) {
        for (std::size_t place = 0; place < plane; ++place) {
            picture.values[place] += weights[static_cast<std::size_t>(map) * plane + place];
        }
    }
    const float scale = largest_magnitude(picture.values);
    if (scale > 0.0f) {
        for (float& value : picture.values) {
            value /= scale;
        }
    }
    return picture;
}

float hidden_strength(const Cnn& model, int neuron) {
    const std::size_t inputs = model.hidden_inputs();
    float strongest = 0.0f;
    for (int other = 0; other < model.hidden_neurons(); ++other) {
        strongest = std::max(strongest, length(model.hidden_weights().subspan(static_cast<std::size_t>(other) * inputs, inputs)));
    }
    return strongest > 0.0f
            ? length(model.hidden_weights().subspan(static_cast<std::size_t>(neuron) * inputs, inputs)) / strongest
            : 0.0f;
}

float hidden_connection(const Cnn& model, int neuron) {
    const std::span<const float> weights = model.output_weights();
    const std::size_t hidden = static_cast<std::size_t>(model.hidden_neurons());
    return weights[static_cast<std::size_t>(ClassId::Second) * hidden + static_cast<std::size_t>(neuron)]
            - weights[static_cast<std::size_t>(ClassId::First) * hidden + static_cast<std::size_t>(neuron)];
}

float hidden_activation(const Cnn::Workspace& workspace, int neuron) {
    return workspace.outputs[Cnn::hidden_relu_layer()][static_cast<std::size_t>(neuron)];
}

FilterPicture output_weights_picture(const Cnn& model, ClassId species) {
    const int neurons = model.hidden_neurons();
    const int columns = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(neurons))));
    const int rows = (neurons + columns - 1) / columns;
    FilterPicture picture;
    picture.kind = FilterPicture::Kind::Signed;
    picture.width = columns;
    picture.height = rows;
    picture.values.assign(static_cast<std::size_t>(columns) * rows, 0.0f);
    const std::span<const float> weights = model.output_weights().subspan(
            static_cast<std::size_t>(species) * static_cast<std::size_t>(neurons), static_cast<std::size_t>(neurons));
    std::copy(weights.begin(), weights.end(), picture.values.begin());
    const float scale = largest_magnitude(picture.values);
    if (scale > 0.0f) {
        for (float& value : picture.values) {
            value /= scale;
        }
    }
    return picture;
}

float class_probability(const Cnn::Workspace& workspace, ClassId species) {
    const std::vector<float>& scores = workspace.outputs[Cnn::output_layer()];
    const float top = std::max(scores[0], scores[1]);
    const float first = std::exp(scores[0] - top);
    const float second = std::exp(scores[1] - top);
    return (species == ClassId::First ? first : second) / (first + second);
}

std::span<const float> activation_map(const Cnn::Workspace& workspace, int block, int filter) {
    const std::size_t plane = static_cast<std::size_t>(Cnn::block_side(block)) * Cnn::block_side(block);
    return std::span<const float>(workspace.outputs[Cnn::relu_layer(block)]).subspan(static_cast<std::size_t>(filter) * plane, plane);
}

float mean_activation(std::span<const float> map) {
    return map.empty() ? 0.0f : std::accumulate(map.begin(), map.end(), 0.0f) / static_cast<float>(map.size());
}

}
