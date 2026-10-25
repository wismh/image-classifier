#pragma once

#include "model/layer.h"

namespace game {

// Every output is a weighted sum of every input plus a bias. Parameters: the weights [outputs][inputs], then one bias
// per output.
class DenseLayer final : public Layer {
public:
    DenseLayer(std::size_t inputs, std::size_t outputs)
        : inputs_(inputs)
        , outputs_(outputs) {}

    [[nodiscard]] std::size_t input_size() const override { return inputs_; }
    [[nodiscard]] std::size_t output_size() const override { return outputs_; }
    [[nodiscard]] std::size_t param_count() const override { return inputs_ * outputs_ + outputs_; }

    void init(std::span<float> params, std::mt19937& rng) const override;
    void forward(std::span<const float> params, std::span<const float> in, std::span<float> out) const override;
    void backward(std::span<const float> params, std::span<const float> in, std::span<const float> out,
            std::span<const float> d_out, std::span<float> d_in, std::span<float> grad_params) const override;

private:
    std::size_t inputs_;
    std::size_t outputs_;
};

}
