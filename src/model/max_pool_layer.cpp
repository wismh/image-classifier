#include "model/max_pool_layer.h"

#include "kernels/layer_ops.h"

namespace game {

void MaxPoolLayer::forward(std::span<const float>, std::span<const float> in, std::span<float> out) const {
    max_pool_forward(in.data(), out.data(), channels_, height_, width_);
}

void MaxPoolLayer::backward(std::span<const float>, std::span<const float> in, std::span<const float>,
        std::span<const float> d_out, std::span<float> d_in, std::span<float>) const {
    if (d_in.empty()) {
        return;
    }
    max_pool_backward(in.data(), d_out.data(), d_in.data(), channels_, height_, width_);
}

}
