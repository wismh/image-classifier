#include "model/adam.h"

#include "kernels/vector_ops.h"

#include <cmath>

namespace game {

namespace {

constexpr float kBeta1 = 0.9f;
constexpr float kBeta2 = 0.999f;
constexpr float kEpsilon = 1e-8f;

}

void Adam::begin_step() {
    ++steps_;
    const auto t = static_cast<float>(steps_);
    scale1_ = 1.0f / (1.0f - std::pow(kBeta1, t));
    scale2_ = 1.0f / (1.0f - std::pow(kBeta2, t));
}

void Adam::update_range(std::span<float> params, std::span<const float> grads, std::size_t begin, std::size_t end,
        float learning_rate, float grad_scale) {
    adam_update(params.data() + begin, grads.data() + begin, first_.data() + begin, second_.data() + begin, end - begin,
            {learning_rate, kBeta1, kBeta2, scale1_, scale2_, kEpsilon, grad_scale});
}

void Adam::step(std::span<float> params, std::span<const float> grads, float learning_rate) {
    begin_step();
    update_range(params, grads, 0, params.size(), learning_rate);
}

}
