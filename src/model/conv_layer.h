#pragma once

#include "model/layer.h"

namespace game {

// A 3x3 convolution with zero padding of 1 and stride 1: the picture keeps its height and width. Parameters: the
// weights [out_channels][in_channels][3][3], then one bias per output channel.
class ConvLayer final : public Layer {
public:
    static constexpr int kKernel = 3;

    ConvLayer(int in_channels, int out_channels, int height, int width)
        : in_channels_(in_channels)
        , out_channels_(out_channels)
        , height_(height)
        , width_(width) {}

    [[nodiscard]] std::size_t input_size() const override { return plane() * static_cast<std::size_t>(in_channels_); }
    [[nodiscard]] std::size_t output_size() const override { return plane() * static_cast<std::size_t>(out_channels_); }
    [[nodiscard]] std::size_t param_count() const override { return weight_count() + static_cast<std::size_t>(out_channels_); }

    void init(std::span<float> params, std::mt19937& rng) const override;
    void forward(std::span<const float> params, std::span<const float> in, std::span<float> out) const override;
    void backward(std::span<const float> params, std::span<const float> in, std::span<const float> out,
            std::span<const float> d_out, std::span<float> d_in, std::span<float> grad_params) const override;

private:
    [[nodiscard]] std::size_t plane() const { return static_cast<std::size_t>(height_) * width_; }
    [[nodiscard]] std::size_t weight_count() const {
        return static_cast<std::size_t>(out_channels_) * in_channels_ * kKernel * kKernel;
    }

    int in_channels_;
    int out_channels_;
    int height_;
    int width_;
};

}
