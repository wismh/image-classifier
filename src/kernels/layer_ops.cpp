#include "kernels/layer_ops.h"

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace game {

namespace {

// The rows and columns of a plane for which a tap shifted by `shift` still lands inside it.
struct Span {
    int begin;
    int end;
};

Span valid(int shift, int size) {
    return {std::max(0, -shift), std::min(size, size - shift)};
}

}

void unroll_3x3(const float* in, int channels, int height, int width, float* columns) {
    const std::size_t plane = static_cast<std::size_t>(height) * width;
    for (int ic = 0; ic < channels; ++ic) {
        const float* source = in + static_cast<std::size_t>(ic) * plane;
        for (int ky = 0; ky < 3; ++ky) {
            const int dy = ky - 1;
            const Span rows = valid(dy, height);
            for (int kx = 0; kx < 3; ++kx) {
                const int dx = kx - 1;
                const Span span = valid(dx, width);
                float* target = columns + (static_cast<std::size_t>(ic) * 9 + ky * 3 + kx) * plane;
                std::memset(target, 0, plane * sizeof(float));
                for (int y = rows.begin; y < rows.end; ++y) {
                    std::memcpy(target + static_cast<std::ptrdiff_t>(y) * width + span.begin,
                            source + static_cast<std::ptrdiff_t>(y + dy) * width + dx + span.begin,
                            static_cast<std::size_t>(span.end - span.begin) * sizeof(float));
                }
            }
        }
    }
}

void fold_3x3(const float* columns, int channels, int height, int width, float* in) {
    const std::size_t plane = static_cast<std::size_t>(height) * width;
    for (int ic = 0; ic < channels; ++ic) {
        float* target = in + static_cast<std::size_t>(ic) * plane;
        for (int ky = 0; ky < 3; ++ky) {
            const int dy = ky - 1;
            const Span rows = valid(dy, height);
            for (int kx = 0; kx < 3; ++kx) {
                const int dx = kx - 1;
                const Span span = valid(dx, width);
                const float* source = columns + (static_cast<std::size_t>(ic) * 9 + ky * 3 + kx) * plane;
                for (int y = rows.begin; y < rows.end; ++y) {
                    const float* from = source + static_cast<std::ptrdiff_t>(y) * width;
                    float* to = target + static_cast<std::ptrdiff_t>(y + dy) * width + dx;
                    for (int x = span.begin; x < span.end; ++x) {
                        to[x] += from[x];
                    }
                }
            }
        }
    }
}

void add_outer_product(const float* d_out, std::size_t outputs, const float* in, std::size_t inputs, float* grad) {
    for (std::size_t o = 0; o < outputs; ++o) {
        const float delta = d_out[o];
        float* row = grad + o * inputs;
        for (std::size_t i = 0; i < inputs; ++i) {
            row[i] += delta * in[i];
        }
    }
}

void relu_forward(const float* in, float* out, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = std::max(0.0f, in[i]);
    }
}

void relu_backward(const float* out, const float* d_out, float* d_in, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        d_in[i] = out[i] > 0.0f ? d_out[i] : 0.0f;
    }
}

void max_pool_forward(const float* in, float* out, int channels, int height, int width) {
    const int out_height = height / 2;
    const int out_width = width / 2;
    for (int c = 0; c < channels; ++c) {
        const float* source = in + static_cast<std::size_t>(c) * height * width;
        float* target = out + static_cast<std::size_t>(c) * out_height * out_width;
        for (int y = 0; y < out_height; ++y) {
            const float* top = source + static_cast<std::ptrdiff_t>(2 * y) * width;
            const float* bottom = top + width;
            for (int x = 0; x < out_width; ++x) {
                target[y * out_width + x] =
                        std::max(std::max(top[2 * x], top[2 * x + 1]), std::max(bottom[2 * x], bottom[2 * x + 1]));
            }
        }
    }
}

void max_pool_backward(const float* in, const float* d_out, float* d_in, int channels, int height, int width) {
    std::fill_n(d_in, static_cast<std::size_t>(channels) * height * width, 0.0f);
    const int out_height = height / 2;
    const int out_width = width / 2;
    for (int c = 0; c < channels; ++c) {
        const float* source = in + static_cast<std::size_t>(c) * height * width;
        float* back = d_in + static_cast<std::size_t>(c) * height * width;
        const float* upstream = d_out + static_cast<std::size_t>(c) * out_height * out_width;
        for (int y = 0; y < out_height; ++y) {
            const float* top = source + static_cast<std::ptrdiff_t>(2 * y) * width;
            const float* bottom = top + width;
            for (int x = 0; x < out_width; ++x) {
                // The first largest, scanning the block row by row.
                const float values[4] = {top[2 * x], top[2 * x + 1], bottom[2 * x], bottom[2 * x + 1]};
                const int which = static_cast<int>(std::max_element(values, values + 4) - values);
                back[static_cast<std::ptrdiff_t>(2 * y + which / 2) * width + 2 * x + which % 2] +=
                        upstream[y * out_width + x];
            }
        }
    }
}

}
