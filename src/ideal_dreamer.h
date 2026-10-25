#pragma once

#include "model/cnn.h"
#include "model/dream.h"
#include "view/ideal_paint.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <vector>

namespace game {

// Works out, in the background, what a network takes for the ideal cat and the ideal dog and hands each picture to the
// paint as soon as it is ready: one task per species, both reading the same network.
class IdealDreamer {
public:
    explicit IdealDreamer(IdealPaint& paint)
        : paint_(paint) {}

    ~IdealDreamer() { stop(); }

    IdealDreamer(const IdealDreamer&) = delete;
    IdealDreamer& operator=(const IdealDreamer&) = delete;

    // Drops the pictures of the previous network and starts on `model`.
    void start(std::shared_ptr<const Cnn> model);
    // Cancels the tasks still running and waits for them.
    void stop();
    // Once per frame on the main thread.
    void collect();

private:
    IdealPaint& paint_;
    std::shared_ptr<std::atomic<bool>> cancel_;
    std::array<std::future<std::optional<std::vector<std::uint8_t>>>, kNumClasses> dreams_;
};

}
