#include "filter_dreamer.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace game {

namespace {

std::optional<std::vector<std::uint8_t>> dream_unit(const Cnn& model, int layer, int unit,
        const std::atomic<bool>& cancel) {
    if (FilterLayers::is_conv(layer)) {
        return dream_filter(model, layer, unit, cancel);
    }
    if (layer == FilterLayers::kHidden) {
        return dream_hidden(model, unit, cancel);
    }
    return dream_class(model, static_cast<ClassId>(unit), cancel);
}

}

void FilterDreamer::reset(std::shared_ptr<const Cnn> model) {
    stop();
    model_ = std::move(model);
    for (int layer = 0; layer < FilterLayers::kCount; ++layer) {
        const auto l = static_cast<std::size_t>(layer);
        batches_[l].reset();
        workers_[l].clear();
        pictures_[l].assign(model_ ? static_cast<std::size_t>(FilterLayers::units(*model_, layer)) : 0, std::nullopt);
        done_[l] = 0;
    }
}

void FilterDreamer::ask(int layer) {
    if (!model_) {
        return;
    }
    for (int other = 0; other < FilterLayers::kCount; ++other) {
        if (other != layer) {
            halt(other);
        }
    }
    const auto l = static_cast<std::size_t>(layer);
    if (batches_[l]) {
        return;  // already at work on it
    }
    collect();
    const std::size_t units = static_cast<std::size_t>(FilterLayers::units(*model_, layer));
    auto batch = std::make_shared<Batch>(units);
    for (std::size_t unit = 0; unit < units; ++unit) {
        if (!pictures_[l][unit]) {
            batch->todo.push_back(unit);
        }
    }
    if (batch->todo.empty()) {
        return;  // every picture is there
    }
    batches_[l] = batch;
    // Leave a thread for the window and the rest of the program.
    const std::size_t cores = std::thread::hardware_concurrency();
    const std::size_t threads = std::clamp<std::size_t>(cores, 2, batch->todo.size() + 1) - 1;
    for (std::size_t t = 0; t < threads; ++t) {
        workers_[l].push_back(std::async(std::launch::async, [model = model_, batch, layer] {
            for (;;) {
                const std::size_t index = batch->next.fetch_add(1);
                if (index >= batch->todo.size() || batch->cancel.load(std::memory_order_relaxed)) {
                    return;
                }
                const std::size_t unit = batch->todo[index];
                auto picture = dream_unit(*model, layer, static_cast<int>(unit), batch->cancel);
                if (!picture) {
                    return;
                }
                batch->slots[unit].picture = std::move(*picture);
                batch->slots[unit].ready.store(true, std::memory_order_release);
            }
        }));
    }
}

void FilterDreamer::halt(int layer) {
    const auto l = static_cast<std::size_t>(layer);
    if (!batches_[l]) {
        return;
    }
    batches_[l]->cancel.store(true);
    for (auto& worker : workers_[l]) {
        if (worker.valid()) {
            worker.wait();  // returns quickly: a task checks the flag every step
        }
    }
    workers_[l].clear();
    collect();
    batches_[l].reset();
}

void FilterDreamer::stop() {
    for (int layer = 0; layer < FilterLayers::kCount; ++layer) {
        halt(layer);
    }
}

bool FilterDreamer::collect() {
    bool any = false;
    for (std::size_t l = 0; l < batches_.size(); ++l) {
        if (!batches_[l]) {
            continue;
        }
        for (std::size_t unit = 0; unit < pictures_[l].size(); ++unit) {
            Slot& slot = batches_[l]->slots[unit];
            if (!pictures_[l][unit] && slot.ready.load(std::memory_order_acquire)) {
                pictures_[l][unit] = std::move(slot.picture);
                ++done_[l];
                any = true;
            }
        }
    }
    return any;
}

}
