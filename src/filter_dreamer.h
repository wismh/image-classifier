#pragma once

#include "filter_layers.h"
#include "model/cnn.h"
#include "model/dream.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <vector>

namespace game {

// Works out, in the background, the picture each unit of a network answers to hardest: a filter, a hidden neuron, or
// the
// output for a species (the ideal cat and dog). A layer's units are worked on together, on several threads, once
// somebody
// asks for that layer, and only the layer asked for last is worked on: the others stop where they are and go on from
// there
// the next time somebody asks for them. Finished pictures are collected on the main thread.
class FilterDreamer {
public:
    FilterDreamer() = default;
    ~FilterDreamer() { stop(); }

    FilterDreamer(const FilterDreamer&) = delete;
    FilterDreamer& operator=(const FilterDreamer&) = delete;

    // Forgets everything and works for `model` from now on. Nothing is computed until ask().
    void reset(std::shared_ptr<const Cnn> model);
    // Works on the units of `layer` that have no picture yet, and stops the work on any other layer.
    void ask(int layer);
    // Cancels the work in progress and waits for it.
    void stop();
    // Once per frame on the main thread: takes in the pictures that have finished. Returns whether there were any.
    bool collect();

    // Whether `layer` is being worked on.
    [[nodiscard]] bool asked(int layer) const { return batches_[static_cast<std::size_t>(layer)] != nullptr; }
    // The picture of a unit (kImageBytes bytes), once it is ready.
    [[nodiscard]] const std::optional<std::vector<std::uint8_t>>& picture(int layer, int unit) const {
        return pictures_[static_cast<std::size_t>(layer)][static_cast<std::size_t>(unit)];
    }
    [[nodiscard]] std::size_t done(int layer) const { return done_[static_cast<std::size_t>(layer)]; }
    [[nodiscard]] std::size_t total(int layer) const {
        return model_ ? static_cast<std::size_t>(FilterLayers::units(*model_, layer)) : 0;
    }

private:
    // One slot per unit of a layer; the tasks write a slot and then say it is ready, the main thread reads it only
    // after.
    struct Slot {
        std::atomic<bool> ready{false};
        std::vector<std::uint8_t> picture;
    };
    struct Batch {
        explicit Batch(std::size_t count)
            : slots(count) {}
        std::vector<Slot> slots;        // one per unit of the layer
        std::vector<std::size_t> todo;  // the units to work out, in order
        std::atomic<std::size_t> next{0};  // the next of `todo`
        std::atomic<bool> cancel{false};
    };

    // Stops the work on a layer, keeps what it finished, and forgets the batch.
    void halt(int layer);

    std::shared_ptr<const Cnn> model_;
    std::array<std::shared_ptr<Batch>, FilterLayers::kCount> batches_;
    std::array<std::vector<std::future<void>>, FilterLayers::kCount> workers_;
    std::array<std::vector<std::optional<std::vector<std::uint8_t>>>, FilterLayers::kCount> pictures_;
    std::array<std::size_t, FilterLayers::kCount> done_{};
};

}
