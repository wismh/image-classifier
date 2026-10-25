#include "model/dream.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>

namespace game {

namespace {

constexpr float kStartNoise = 0.05f;

constexpr std::size_t kPlane = static_cast<std::size_t>(kImageSide) * kImageSide;

// Moves the picture by (dx, dy), what leaves on one side coming back on the other: `to` gets from[y - dy][x - dx].
void roll(std::span<const float> from, std::vector<float>& to, int dx, int dy) {
    for (int c = 0; c < kChannels; ++c) {
        for (int y = 0; y < kImageSide; ++y) {
            const int source_y = ((y - dy) % kImageSide + kImageSide) % kImageSide;
            for (int x = 0; x < kImageSide; ++x) {
                const int source_x = ((x - dx) % kImageSide + kImageSide) % kImageSide;
                to[c * kPlane + static_cast<std::size_t>(y) * kImageSide + x] =
                        from[c * kPlane + static_cast<std::size_t>(source_y) * kImageSide + source_x];
            }
        }
    }
}

void blur(std::vector<float>& image, float amount) {
    const std::vector<float> source = image;
    for (int c = 0; c < kChannels; ++c) {
        for (int y = 0; y < kImageSide; ++y) {
            for (int x = 0; x < kImageSide; ++x) {
                float sum = 0.0f;
                int count = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = x + dx;
                        const int ny = y + dy;
                        if (nx >= 0 && nx < kImageSide && ny >= 0 && ny < kImageSide) {
                            sum += source[c * kPlane + static_cast<std::size_t>(ny) * kImageSide + nx];
                            ++count;
                        }
                    }
                }
                float& value = image[c * kPlane + static_cast<std::size_t>(y) * kImageSide + x];
                value += amount * (sum / static_cast<float>(count) - value);
            }
        }
    }
}

// What a search wants of the network: run it on `input` and leave the gradient of the thing to make smaller in
// `workspace.input_gradient()`.
using Objective = std::function<void(const Cnn&, Cnn::Workspace& workspace, std::span<const float> input)>;

std::optional<std::vector<std::uint8_t>> descend(const Cnn& model, std::uint32_t seed, const std::atomic<bool>& cancel,
        const DreamSettings& settings, const Objective& objective) {
    std::mt19937 rng{seed};
    std::uniform_real_distribution<float> noise(-kStartNoise, kStartNoise);
    std::uniform_int_distribution<int> shift(-settings.jitter, settings.jitter);

    std::vector<float> image(Cnn::kInputSize);
    for (float& value : image) {
        value = noise(rng);
    }
    std::vector<float> shifted(Cnn::kInputSize);
    std::vector<float> gradient(Cnn::kInputSize);
    Cnn::Workspace workspace = model.make_workspace(true);

    for (int step = 0; step < settings.steps; ++step) {
        if (cancel.load(std::memory_order_relaxed)) {
            return std::nullopt;
        }
        const int dx = shift(rng);
        const int dy = shift(rng);
        roll(image, shifted, dx, dy);
        objective(model, workspace, shifted);
        roll(workspace.input_gradient(), gradient, -dx, -dy);

        float largest = 0.0f;
        for (const float g : gradient) {
            largest = std::max(largest, std::abs(g));
        }
        if (largest == 0.0f && step == 0) {
            // Nothing under the goal answers to faint noise (every unit is silent): start from a stronger picture.
            std::uniform_real_distribution<float> loud(-0.3f, 0.3f);
            for (float& value : image) {
                value = loud(rng);
            }
            continue;
        }
        const float t = static_cast<float>(step) / static_cast<float>(std::max(1, settings.steps - 1));
        const float rate = settings.first_step + (settings.last_step - settings.first_step) * t;
        const float scale = largest > 0.0f ? rate / largest : 0.0f;
        for (std::size_t i = 0; i < image.size(); ++i) {
            image[i] = std::clamp(image[i] * (1.0f - settings.decay) - scale * gradient[i], -0.5f, 0.5f);
        }
        blur(image, settings.blur);
    }

    // Back from the network's input to bytes, as a picture of a dataset: channel-major floats in [-0.5, 0.5] to rows of
    // RGB bytes.
    std::vector<std::uint8_t> picture(kImageBytes);
    for (std::size_t i = 0; i < kPlane; ++i) {
        for (int c = 0; c < kChannels; ++c) {
            picture[i * kChannels + c] =
                    static_cast<std::uint8_t>(std::lround(std::clamp(image[c * kPlane + i] + 0.5f, 0.0f, 1.0f) * 255.0f));
        }
    }
    return picture;
}

}

std::optional<std::vector<std::uint8_t>> dream_class(const Cnn& model, ClassId species,
        const std::atomic<bool>& cancel, const DreamSettings& settings) {
    // Loss = the other species' score minus this species' score: it falls as the network grows surer, and unlike the
    // loss of the probability its gradient does not fade away when the network is already certain.
    std::vector<float> d_scores(kNumClasses, 0.0f);
    d_scores[static_cast<std::size_t>(species)] = -1.0f;
    d_scores[1 - static_cast<std::size_t>(species)] = 1.0f;
    // The layers add the gradients of their parameters to this; the search wants only the picture's, so it is never
    // read.
    std::vector<float> unused(model.param_count());
    return descend(model, static_cast<std::uint32_t>(species) + 1u, cancel, settings,
            [&](const Cnn& cnn, Cnn::Workspace& workspace, std::span<const float> input) {
                (void)cnn.forward(input, workspace);
                cnn.backward(input, workspace, d_scores, unused);
            });
}

std::optional<std::vector<std::uint8_t>> dream_hidden(const Cnn& model, int neuron, const std::atomic<bool>& cancel,
        const DreamSettings& settings) {
    // As for a filter, on the neuron's output before its ReLU.
    const std::size_t layer = Cnn::hidden_layer();
    std::vector<float> d_output(static_cast<std::size_t>(model.hidden_neurons()), 0.0f);
    d_output[static_cast<std::size_t>(neuron)] = -1.0f;
    // The layers add the gradients of their parameters to this; the search wants only the picture's, so it is never
    // read.
    std::vector<float> unused(model.param_count());
    const auto seed = static_cast<std::uint32_t>(900000 + neuron);
    return descend(model, seed, cancel, settings,
            [&](const Cnn& cnn, Cnn::Workspace& workspace, std::span<const float> input) {
                (void)cnn.forward_until(input, workspace, layer);
                cnn.backward_until(input, workspace, layer, d_output, unused);
            });
}

std::optional<std::vector<std::uint8_t>> dream_filter(const Cnn& model, int block, int filter,
        const std::atomic<bool>& cancel, const DreamSettings& settings) {
    // Loss = minus the mean output of the filter's convolution, so that the search makes the filter answer as strongly
    // as
    // it can everywhere. The output before the ReLU is used: a filter that is silent for noise still has a gradient
    // there.
    const std::size_t layer = Cnn::conv_layer(block);
    const std::size_t plane = static_cast<std::size_t>(Cnn::block_side(block)) * Cnn::block_side(block);
    std::vector<float> d_output(static_cast<std::size_t>(model.filters(block)) * plane, 0.0f);
    std::fill_n(d_output.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(filter) * plane), plane,
            -1.0f / static_cast<float>(plane));
    // The layers add the gradients of their parameters to this; the search wants only the picture's, so it is never
    // read.
    std::vector<float> unused(model.param_count());
    const auto seed = static_cast<std::uint32_t>(1000 + block * 4096 + filter);
    return descend(model, seed, cancel, settings,
            [&](const Cnn& cnn, Cnn::Workspace& workspace, std::span<const float> input) {
                (void)cnn.forward_until(input, workspace, layer);
                cnn.backward_until(input, workspace, layer, d_output, unused);
            });
}

}
