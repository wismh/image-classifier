#pragma once

#include "kernels/layer_ops.h"
#include "model/layer.h"

namespace game {

// max(0, x) on every value.
class ReluLayer final : public Layer {
public:
    explicit ReluLayer(std::size_t size)
        : size_(size) {}

    [[nodiscard]] std::size_t input_size() const override { return size_; }
    [[nodiscard]] std::size_t output_size() const override { return size_; }

    void forward(std::span<const float>, std::span<const float> in, std::span<float> out) const override {
        relu_forward(in.data(), out.data(), size_);
    }

    void backward(std::span<const float>, std::span<const float>, std::span<const float> out,
            std::span<const float> d_out, std::span<float> d_in, std::span<float>) const override {
        if (d_in.empty()) {
            return;
        }
        relu_backward(out.data(), d_out.data(), d_in.data(), size_);
    }

private:
    std::size_t size_;
};

}
