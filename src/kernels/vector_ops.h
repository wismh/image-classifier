#pragma once

#include <cstddef>

namespace game {

// dst[i] += src[i].
void add_floats(float* dst, const float* src, std::size_t count);

struct AdamSettings {
    float learning_rate;
    float beta1;
    float beta2;
    float scale1;      // 1 / (1 - beta1^t)
    float scale2;      // 1 / (1 - beta2^t)
    float epsilon;
    float grad_scale;  // every gradient is multiplied by this first
};

// One Adam step over `count` parameters, with their running means `first` and `second`.
void adam_update(float* params, const float* grads, float* first, float* second, std::size_t count,
        const AdamSettings& settings);

}
