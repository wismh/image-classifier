#pragma once

#include "model/layer.h"

namespace game {

// The largest value of every 2x2 block of every channel: height and width halve. Height and width must be even. The
// gradient goes back to the first largest value of the block, found again from the input, so nothing is stored.
class MaxPoolLayer final : public Layer {
public:
    MaxPoolLayer(int channels, int height, int width)
        : channels_(channels)
        , height_(height)
        , width_(width) {}

    [[nodiscard]] std::size_t input_size() const override {
        return static_cast<std::size_t>(channels_) * height_ * width_;
    }
    [[nodiscard]] std::size_t output_size() const override { return input_size() / 4; }

    void forward(std::span<const float> params, std::span<const float> in, std::span<float> out) const override;
    void backward(std::span<const float> params, std::span<const float> in, std::span<const float> out,
            std::span<const float> d_out, std::span<float> d_in, std::span<float> grad_params) const override;

private:
    int channels_;
    int height_;
    int width_;
};

}
