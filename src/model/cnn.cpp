#include "model/cnn.h"

#include "model/conv_layer.h"
#include "model/dense_layer.h"
#include "model/max_pool_layer.h"
#include "model/relu_layer.h"

#include <format>
#include <random>

namespace game {

namespace {

std::vector<std::shared_ptr<const Layer>> build_layers(const CnnConfig& config) {
    std::vector<std::shared_ptr<const Layer>> layers;
    int channels = kChannels;
    int side = kImageSide;
    for (const int filters : {config.conv1, config.conv2, config.conv3}) {
        layers.push_back(std::make_shared<ConvLayer>(channels, filters, side, side));
        layers.push_back(std::make_shared<ReluLayer>(static_cast<std::size_t>(filters) * side * side));
        layers.push_back(std::make_shared<MaxPoolLayer>(filters, side, side));
        channels = filters;
        side /= 2;
    }
    const auto flat = static_cast<std::size_t>(channels) * side * side;
    layers.push_back(std::make_shared<DenseLayer>(flat, static_cast<std::size_t>(config.hidden)));
    layers.push_back(std::make_shared<ReluLayer>(static_cast<std::size_t>(config.hidden)));
    layers.push_back(std::make_shared<DenseLayer>(static_cast<std::size_t>(config.hidden), kNumClasses));
    return layers;
}

}

Cnn::Cnn(const CnnConfig& config, std::vector<float> params)
    : config_(config)
    , layers_(build_layers(config)) {
    std::size_t total = 0;
    for (const auto& layer : layers_) {
        offsets_.push_back(total);
        total += layer->param_count();
    }
    params_ = std::move(params);
    params_.resize(total);
}

Cnn::Cnn(const CnnConfig& config, std::uint32_t seed)
    : Cnn(config, std::vector<float>{}) {
    std::mt19937 rng{seed};
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        layers_[i]->init(std::span<float>(params_).subspan(offsets_[i], layers_[i]->param_count()), rng);
    }
}

std::expected<Cnn, std::string> Cnn::with_params(const CnnConfig& config, std::vector<float> params) {
    if (!config.valid()) {
        return std::unexpected("the network size is out of range");
    }
    const std::size_t given = params.size();
    Cnn cnn{config, std::move(params)};
    if (cnn.params_.size() != given) {
        return std::unexpected(std::format("expected {} values for network {}, got {}", cnn.params_.size(),
                config.label(), given));
    }
    return cnn;
}

Cnn::Workspace Cnn::make_workspace(bool input_gradient) const {
    Workspace workspace;
    for (const auto& layer : layers_) {
        workspace.outputs.emplace_back(layer->output_size());
        const bool first = workspace.deltas.empty();
        workspace.deltas.emplace_back(first && !input_gradient ? 0 : layer->input_size());
    }
    return workspace;
}

int Cnn::filters(int block) const {
    return block == 0 ? config_.conv1 : block == 1 ? config_.conv2 : config_.conv3;
}

std::span<const float> Cnn::conv_weights(int block) const {
    const std::size_t count = static_cast<std::size_t>(filters(block)) * block_inputs(block) * 9;
    return std::span<const float>(params_).subspan(offsets_[conv_layer(block)], count);
}

float Cnn::conv_bias(int block, int filter) const {
    const std::size_t count = static_cast<std::size_t>(filters(block)) * block_inputs(block) * 9;
    return params_[offsets_[conv_layer(block)] + count + static_cast<std::size_t>(filter)];
}

std::size_t Cnn::hidden_inputs() const {
    const std::size_t side = static_cast<std::size_t>(block_side(kBlocks - 1)) / 2;  // the last block pools its maps
    return static_cast<std::size_t>(filters(kBlocks - 1)) * side * side;
}

std::span<const float> Cnn::hidden_weights() const {
    return std::span<const float>(params_).subspan(offsets_[hidden_layer()],
            hidden_inputs() * static_cast<std::size_t>(hidden_neurons()));
}

float Cnn::hidden_bias(int neuron) const {
    return params_[offsets_[hidden_layer()] + hidden_inputs() * static_cast<std::size_t>(hidden_neurons())
            + static_cast<std::size_t>(neuron)];
}

std::span<const float> Cnn::output_weights() const {
    return std::span<const float>(params_).subspan(offsets_[output_layer()],
            static_cast<std::size_t>(hidden_neurons()) * kNumClasses);
}

float Cnn::output_bias(int species) const {
    return params_[offsets_[output_layer()] + static_cast<std::size_t>(hidden_neurons()) * kNumClasses
            + static_cast<std::size_t>(species)];
}

std::span<const float> Cnn::forward(std::span<const float> input, Workspace& workspace) const {
    return forward_until(input, workspace, layers_.size() - 1);
}

std::span<const float> Cnn::forward_until(std::span<const float> input, Workspace& workspace,
        std::size_t last) const {
    std::span<const float> current = input;
    for (std::size_t i = 0; i <= last; ++i) {
        layers_[i]->forward(std::span<const float>(params_).subspan(offsets_[i], layers_[i]->param_count()), current,
                workspace.outputs[i]);
        current = workspace.outputs[i];
    }
    return current;
}

void Cnn::backward(std::span<const float> input, Workspace& workspace, std::span<const float> d_scores,
        std::span<float> grads) const {
    backward_until(input, workspace, layers_.size() - 1, d_scores, grads);
}

void Cnn::backward_until(std::span<const float> input, Workspace& workspace, std::size_t last,
        std::span<const float> d_output, std::span<float> grads) const {
    std::span<const float> upstream = d_output;
    for (std::size_t i = last + 1; i-- > 0;) {
        const std::span<const float> layer_input = i == 0 ? input : std::span<const float>(workspace.outputs[i - 1]);
        const std::size_t count = layers_[i]->param_count();
        layers_[i]->backward(std::span<const float>(params_).subspan(offsets_[i], count), layer_input,
                workspace.outputs[i], upstream, workspace.deltas[i], grads.subspan(offsets_[i], count));
        upstream = workspace.deltas[i];
    }
}

}
