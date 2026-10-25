#pragma once

#include "model/cnn.h"

#include <atomic>
#include <cstdint>
#include <optional>
#include <vector>

namespace game {

// How the search for an ideal picture goes. The values are in the network's input range, which is 1 wide.
struct DreamSettings {
    // Chosen by looking at what several settings made of a trained network: with a stronger step and a lighter blur the
    // pictures keep sharp features (eyes, noses, pointed ears) that softer settings
    // smear into a gray haze.
    int steps = 400;
    float first_step = 0.12f;  // the largest change of one value in the first step
    float last_step = 0.01f;   // and in the last: the steps shrink evenly
    // Per step, every value shrinks by this share toward the middle gray, which keeps the picture to what the species
    // needs.
    float decay = 0.002f;
    // Per step, every value moves this far toward the mean of its 3x3 neighbourhood.
    float blur = 0.12f;
    int jitter = 4;  // the picture is shifted by up to this many pixels each way before every step
};

// What the network takes for the ideal `species`: a picture found by changing faint noise, step by step, so that the
// network's score for the species rises over its score for the other one. Every step is a small move against the
// gradient with respect to the picture, taken on a randomly shifted copy (so that the result does not hang on exact
// pixel positions) and followed by a light blur (so that it comes out as shapes and colors rather than speckle).
// The result is a normalized picture (kImageBytes bytes), the very picture the network was given: nothing is stretched
// or
// recolored afterwards. Reads the model only, so it can run on any thread. Returns nullopt when `cancel` was set.
[[nodiscard]] std::optional<std::vector<std::uint8_t>> dream_class(const Cnn& model, ClassId species,
        const std::atomic<bool>& cancel, const DreamSettings& settings = {});

// The same for one filter of the network's convolutions: the picture it answers to hardest, on average over the whole
// picture. `block` is 0..Cnn::kBlocks-1, `filter` below model.filters(block). Deeper filters read the filters before
// them, so what comes out is a texture or a part of something rather than a whole animal.
[[nodiscard]] std::optional<std::vector<std::uint8_t>> dream_filter(const Cnn& model, int block, int filter,
        const std::atomic<bool>& cancel, const DreamSettings& settings = {});

// And for one neuron of the hidden layer: the picture it answers to hardest. It reads the whole picture at once, so the
// result is a picture of whatever the neuron is a detector of.
[[nodiscard]] std::optional<std::vector<std::uint8_t>> dream_hidden(const Cnn& model, int neuron,
        const std::atomic<bool>& cancel, const DreamSettings& settings = {});

}
