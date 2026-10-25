#include "kernels/vector_ops.h"

#include <cmath>

namespace game {

void add_floats(float* dst, const float* src, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        dst[i] += src[i];
    }
}

void adam_update(float* params, const float* grads, float* first, float* second, std::size_t count,
        const AdamSettings& s) {
    for (std::size_t i = 0; i < count; ++i) {
        const float grad = grads[i] * s.grad_scale;
        first[i] = s.beta1 * first[i] + (1.0f - s.beta1) * grad;
        second[i] = s.beta2 * second[i] + (1.0f - s.beta2) * grad * grad;
        params[i] -= s.learning_rate * (first[i] * s.scale1) / (std::sqrt(second[i] * s.scale2) + s.epsilon);
    }
}

}
