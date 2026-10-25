#pragma once

#include <cstddef>

namespace game {

// The loops of the layers that touch every value of an activation: unrolling for the convolution, ReLU, pooling, the
// gradient of a dense layer. They live in the kernel library next to the matrix products so that they are compiled with
// optimization in every configuration. Pictures are channel-major: [channels][height][width].

// Unrolls a picture for a 3x3 convolution with zero padding of 1: `columns` is [channels * 9][height * width], and the
// row of tap (channel, ky, kx) holds, for every position, the value that tap multiplies (zero outside the picture).
void unroll_3x3(const float* in, int channels, int height, int width, float* columns);
// The reverse: ADDS every value of `columns` to the input value it was taken from.
void fold_3x3(const float* columns, int channels, int height, int width, float* in);

// grad[o][i] += d_out[o] * in[i] for every output o and input i; `grad` is [outputs][inputs].
void add_outer_product(const float* d_out, std::size_t outputs, const float* in, std::size_t inputs, float* grad);

void relu_forward(const float* in, float* out, std::size_t count);
// d_in = d_out where out > 0, else 0.
void relu_backward(const float* out, const float* d_out, float* d_in, std::size_t count);

// The largest of every 2x2 block of every channel; height and width must be even.
void max_pool_forward(const float* in, float* out, int channels, int height, int width);
// Sends each gradient to the first largest value of its block (found again from `in`); `d_in` is overwritten.
void max_pool_backward(const float* in, const float* d_out, float* d_in, int channels, int height, int width);

}
