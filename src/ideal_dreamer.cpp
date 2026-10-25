#include "ideal_dreamer.h"

#include <chrono>

namespace game {

void IdealDreamer::start(std::shared_ptr<const Cnn> model) {
    stop();
    paint_.reset();
    cancel_ = std::make_shared<std::atomic<bool>>(false);
    for (std::size_t species = 0; species < kNumClasses; ++species) {
        dreams_[species] = std::async(std::launch::async, [model, cancel = cancel_, species] {
            return dream_class(*model, static_cast<ClassId>(species), *cancel);
        });
    }
}

void IdealDreamer::stop() {
    if (cancel_) {
        cancel_->store(true);
    }
    for (auto& dream : dreams_) {
        if (dream.valid()) {
            dream.wait();  // returns quickly: the task checks the flag every step
            dream = {};
        }
    }
}

void IdealDreamer::collect() {
    for (std::size_t species = 0; species < kNumClasses; ++species) {
        auto& dream = dreams_[species];
        if (dream.valid() && dream.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            if (const auto picture = dream.get()) {
                paint_.set(static_cast<ClassId>(species), *picture);
            }
        }
    }
}

}
