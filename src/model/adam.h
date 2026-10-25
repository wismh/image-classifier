#pragma once

#include <span>
#include <vector>

namespace game {

// Adam: a step per parameter scaled by a running mean of its gradient over a running mean of its squared size, so the
// parameters of every layer move at a comparable rate whatever the scale of their gradients. A step can be taken in
// pieces on several threads: begin_step() once, then update_range() for pieces that do not overlap.
class Adam {
public:
    explicit Adam(std::size_t param_count)
        : first_(param_count, 0.0f)
        , second_(param_count, 0.0f) {}

    // Starts a step: the pieces that follow belong to it.
    void begin_step();
    // Moves params[begin, end) against `grads` (indexed like `params`, each multiplied by `grad_scale`). Pieces that do
    // not overlap may run at the same time.
    void update_range(std::span<float> params, std::span<const float> grads, std::size_t begin, std::size_t end,
            float learning_rate, float grad_scale = 1.0f);
    // The whole step at once.
    void step(std::span<float> params, std::span<const float> grads, float learning_rate);

private:
    std::vector<float> first_;
    std::vector<float> second_;
    unsigned long long steps_ = 0;
    // Corrections of the running means for starting at zero, for the current step.
    float scale1_ = 1.0f;
    float scale2_ = 1.0f;
};

}
