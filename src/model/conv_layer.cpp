#include "model/conv_layer.h"

#include "kernels/gemm.h"
#include "kernels/layer_ops.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace game {

namespace {

// Working space of one call, kept per thread so that a layer shared by every thread needs no locks and a call allocates
// nothing after the first.
struct Scratch {
    std::vector<float> columns;       // the unrolled input: [in_channels * 9][height * width]
    std::vector<float> column_grads;  // the same shape, the gradient with respect to it
    std::vector<float> weights_t;     // the weights turned over: [in_channels * 9][out_channels]
};

Scratch& scratch() {
    thread_local Scratch instance;
    return instance;
}

float* grown(std::vector<float>& buffer, std::size_t size) {
    if (buffer.size() < size) {
        buffer.resize(size);
    }
    return buffer.data();
}

}

void ConvLayer::init(std::span<float> params, std::mt19937& rng) const {
    // He initialization: the spread keeps the size of the activations steady through ReLU layers.
    const float deviation = std::sqrt(2.0f / static_cast<float>(in_channels_ * kKernel * kKernel));
    std::normal_distribution<float> normal{0.0f, deviation};
    std::fill(params.begin(), params.end(), 0.0f);
    for (std::size_t i = 0; i < weight_count(); ++i) {
        params[i] = normal(rng);
    }
}

void ConvLayer::forward(std::span<const float> params, std::span<const float> in, std::span<float> out) const {
    const int taps = in_channels_ * kKernel * kKernel;
    const int positions = height_ * width_;
    float* columns = grown(scratch().columns, static_cast<std::size_t>(taps) * positions);
    unroll_3x3(in.data(), in_channels_, height_, width_, columns);

    // The biases first, then the product adds to them.
    const float* biases = params.data() + weight_count();
    for (int oc = 0; oc < out_channels_; ++oc) {
        std::fill_n(out.data() + static_cast<std::size_t>(oc) * positions, positions, biases[oc]);
    }
    best_gemm().nn(out_channels_, positions, taps, params.data(), taps, columns, positions, out.data(), positions, true);
}

void ConvLayer::backward(std::span<const float> params, std::span<const float> in, std::span<const float>,
        std::span<const float> d_out, std::span<float> d_in, std::span<float> grad_params) const {
    const int taps = in_channels_ * kKernel * kKernel;
    const int positions = height_ * width_;
    const GemmKernels& gemm = best_gemm();
    Scratch& work = scratch();

    float* grad_biases = grad_params.data() + weight_count();
    for (int oc = 0; oc < out_channels_; ++oc) {
        const float* delta = d_out.data() + static_cast<std::size_t>(oc) * positions;
        float sum = 0.0f;
        for (int i = 0; i < positions; ++i) {
            sum += delta[i];
        }
        grad_biases[oc] += sum;
    }

    // Weights: every gradient is the dot product of a row of d_out with the row of `columns` its tap multiplied.
    float* columns = grown(work.columns, static_cast<std::size_t>(taps) * positions);
    unroll_3x3(in.data(), in_channels_, height_, width_, columns);
    gemm.nt(out_channels_, taps, positions, d_out.data(), positions, columns, positions, grad_params.data(), taps, true);

    if (d_in.empty()) {
        return;
    }
    // Input: the gradient of every unrolled value is the weights turned over times d_out; folding adds each back to the
    // input position it was taken from.
    float* weights_t = grown(work.weights_t, static_cast<std::size_t>(taps) * out_channels_);
    for (int oc = 0; oc < out_channels_; ++oc) {
        for (int t = 0; t < taps; ++t) {
            weights_t[static_cast<std::size_t>(t) * out_channels_ + oc] = params[static_cast<std::size_t>(oc) * taps + t];
        }
    }
    float* column_grads = grown(work.column_grads, static_cast<std::size_t>(taps) * positions);
    gemm.nn(taps, positions, out_channels_, weights_t, out_channels_, d_out.data(), positions, column_grads, positions,
            false);
    std::fill(d_in.begin(), d_in.end(), 0.0f);
    fold_3x3(column_grads, in_channels_, height_, width_, d_in.data());
}

}
