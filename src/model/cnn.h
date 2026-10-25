#pragma once

#include "model/cnn_config.h"
#include "model/layer.h"
#include "model/pet_image.h"

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace game {

// The convolutional network. It holds the layers and all of its numbers as one parameter vector; running it needs a
// Workspace, one per thread, so the same network serves every thread at once.
class Cnn {
public:
    // Input: a picture as kChannels x kImageSide x kImageSide floats, channel-major.
    static constexpr std::size_t kInputSize = static_cast<std::size_t>(kChannels) * kImageSide * kImageSide;

    // What a pass through the network leaves behind: the output of every layer, and the gradient of the loss with
    // respect to the input of every layer (of the first only when asked for in make_workspace()).
    struct Workspace {
        std::vector<std::vector<float>> outputs;
        std::vector<std::vector<float>> deltas;

        // The gradient of the loss with respect to the network's input; only after backward() on a workspace made with
        // `input_gradient`.
        [[nodiscard]] std::span<const float> input_gradient() const { return deltas.front(); }
    };

    // The network is three blocks, each a convolution, a ReLU and a 2x2 pooling, then the dense layers. Block b has
    // filters(b) filters of 3x3 over the block's inputs (3 colors for the first block, the filters of the block before
    // for the others), and its convolution makes planes of block_side(b) x block_side(b) (64, 32, 16).
    static constexpr int kBlocks = 3;
    static constexpr std::size_t kLayersPerBlock = 3;
    [[nodiscard]] static constexpr std::size_t conv_layer(int block) { return static_cast<std::size_t>(block) * kLayersPerBlock; }
    [[nodiscard]] static constexpr std::size_t relu_layer(int block) { return conv_layer(block) + 1; }
    [[nodiscard]] static constexpr int block_side(int block) { return kImageSide >> block; }
    [[nodiscard]] int filters(int block) const;
    [[nodiscard]] int block_inputs(int block) const { return block == 0 ? kChannels : filters(block - 1); }
    // The weights of the convolution of `block`: [filters][inputs][3][3]; and the bias of one of its filters.
    [[nodiscard]] std::span<const float> conv_weights(int block) const;
    [[nodiscard]] float conv_bias(int block, int filter) const;

    // After the blocks come a dense layer of `hidden_neurons()` neurons over the last block's maps (flattened), its
    // ReLU,
    // and the dense layer of the two species' scores. Layer indices as for conv_layer().
    [[nodiscard]] static constexpr std::size_t hidden_layer() { return kBlocks * kLayersPerBlock; }
    [[nodiscard]] static constexpr std::size_t hidden_relu_layer() { return hidden_layer() + 1; }
    [[nodiscard]] static constexpr std::size_t output_layer() { return hidden_layer() + 2; }
    [[nodiscard]] int hidden_neurons() const { return config_.hidden; }
    // What the hidden layer reads: the maps of the last block, one after another: [filters(2)][side][side].
    [[nodiscard]] std::size_t hidden_inputs() const;
    // The weights of the hidden layer, [neurons][hidden_inputs()], and the bias of one neuron.
    [[nodiscard]] std::span<const float> hidden_weights() const;
    [[nodiscard]] float hidden_bias(int neuron) const;
    // The weights of the output layer, [kNumClasses][hidden_neurons()] (the species' numbers as in ClassId), and the
    // bias
    // of one species' score.
    [[nodiscard]] std::span<const float> output_weights() const;
    [[nodiscard]] float output_bias(int species) const;

    // Fresh random weights; the same `seed` gives the same ones. `config` must be valid.
    Cnn(const CnnConfig& config, std::uint32_t seed);
    // Weights from a file. Fails when `params` is not exactly the size of the network `config` describes.
    [[nodiscard]] static std::expected<Cnn, std::string> with_params(const CnnConfig& config, std::vector<float> params);

    [[nodiscard]] const CnnConfig& config() const { return config_; }
    [[nodiscard]] std::size_t param_count() const { return params_.size(); }
    [[nodiscard]] std::span<const float> params() const { return params_; }
    [[nodiscard]] std::span<float> params() { return params_; }

    // With `input_gradient`, backward() also works out the gradient with respect to the input picture, which training
    // does not need and the search for an ideal picture does.
    [[nodiscard]] Workspace make_workspace(bool input_gradient = false) const;
    // Returns the scores of the classes, a view into `workspace` valid until its next use.
    [[nodiscard]] std::span<const float> forward(std::span<const float> input, Workspace& workspace) const;
    // Runs the layers up to and including `last` only (a layer index, see conv_layer()); returns that layer's output, a
    // view into `workspace`.
    [[nodiscard]] std::span<const float> forward_until(std::span<const float> input, Workspace& workspace,
            std::size_t last) const;
    // Call right after forward() on the same input and workspace. ADDS the gradient of the loss with respect to every
    // parameter to `grads` (param_count() values); `d_scores` is the gradient with respect to the scores.
    void backward(std::span<const float> input, Workspace& workspace, std::span<const float> d_scores,
            std::span<float> grads) const;
    // The same for a pass that stopped at layer `last` (forward_until): `d_output` is the gradient with respect to that
    // layer's output.
    void backward_until(std::span<const float> input, Workspace& workspace, std::size_t last,
            std::span<const float> d_output, std::span<float> grads) const;

private:
    Cnn(const CnnConfig& config, std::vector<float> params);

    CnnConfig config_;
    std::vector<std::shared_ptr<const Layer>> layers_;
    std::vector<std::size_t> offsets_;  // where each layer's parameters start
    std::vector<float> params_;
};

}
