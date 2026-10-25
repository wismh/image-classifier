#pragma once

#include <cstddef>
#include <random>
#include <span>

namespace game {

// One step of the network. A layer owns no numbers: its weights are a slice of the network's parameter vector, handed
// in on every call, so one layer object serves every thread and the parameters can be saved and updated as one block.
// Activations are flat float arrays; pictures are channel-major (channel, row, column).
class Layer {
public:
    virtual ~Layer() = default;

    [[nodiscard]] virtual std::size_t input_size() const = 0;
    [[nodiscard]] virtual std::size_t output_size() const = 0;
    [[nodiscard]] virtual std::size_t param_count() const { return 0; }

    // Fills this layer's `params` with the starting weights.
    virtual void init(std::span<float> params, std::mt19937& rng) const { (void)params; (void)rng; }

    virtual void forward(std::span<const float> params, std::span<const float> in, std::span<float> out) const = 0;

    // `d_out` is the gradient of the loss with respect to the output. Writes the gradient with respect to the input
    // into `d_in` (overwriting it; an empty `d_in` means the caller does not need it) and ADDS the gradient with
    // respect to the parameters to `grad_params`. `out` is what forward produced for `in`.
    virtual void backward(std::span<const float> params, std::span<const float> in, std::span<const float> out,
            std::span<const float> d_out, std::span<float> d_in, std::span<float> grad_params) const = 0;
};

}
