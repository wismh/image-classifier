#include "model/dense_layer.h"

#include "kernels/gemm.h"
#include "kernels/layer_ops.h"

#include <algorithm>
#include <cmath>

namespace game {

void DenseLayer::init(std::span<float> params, std::mt19937& rng) const {
    std::normal_distribution<float> normal{0.0f, std::sqrt(2.0f / static_cast<float>(inputs_))};
    std::fill(params.begin(), params.end(), 0.0f);
    for (std::size_t i = 0; i < inputs_ * outputs_; ++i) {
        params[i] = normal(rng);
    }
}

void DenseLayer::forward(std::span<const float> params, std::span<const float> in, std::span<float> out) const {
    const float* biases = params.data() + inputs_ * outputs_;
    std::copy_n(biases, outputs_, out.data());
    // out[o] += dot(weights row o, in): one row of A (the input) against `outputs` rows of B (the weights).
    best_gemm().nt(1, static_cast<int>(outputs_), static_cast<int>(inputs_), in.data(), static_cast<int>(inputs_),
            params.data(), static_cast<int>(inputs_), out.data(), static_cast<int>(outputs_), true);
}

void DenseLayer::backward(std::span<const float> params, std::span<const float> in, std::span<const float>,
        std::span<const float> d_out, std::span<float> d_in, std::span<float> grad_params) const {
    float* grad_biases = grad_params.data() + inputs_ * outputs_;
    for (std::size_t o = 0; o < outputs_; ++o) {
        grad_biases[o] += d_out[o];
    }
    add_outer_product(d_out.data(), outputs_, in.data(), inputs_, grad_params.data());
    if (d_in.empty()) {
        return;
    }
    // d_in[i] = sum over o of weights[o][i] * d_out[o]: one row of A (d_out) times the weights.
    best_gemm().nn(1, static_cast<int>(inputs_), static_cast<int>(outputs_), d_out.data(), static_cast<int>(outputs_),
            params.data(), static_cast<int>(inputs_), d_in.data(), static_cast<int>(inputs_), false);
}

}
