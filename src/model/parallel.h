#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>

namespace game {

// How the domain code runs a loop on several threads without knowing the engine's job system.
// `run(count, grain, fn)` calls `fn(begin, end)` over half-open ranges of at most `grain` indices and returns once
// all are done. `fn` must not throw and may only write what is private to its range.
struct Parallel {
    using RangeFn = std::function<void(std::size_t, std::size_t)>;

    // Threads working at once, the caller included.
    std::size_t workers = 1;
    std::function<void(std::size_t count, std::size_t grain, const RangeFn&)> run;

    // Ranges that give each worker about `per_worker` pieces.
    [[nodiscard]] std::size_t grain_for(std::size_t count, std::size_t per_worker = 1) const {
        const std::size_t pieces = std::max<std::size_t>(1, workers * per_worker);
        return std::max<std::size_t>(1, (count + pieces - 1) / pieces);
    }

    void for_each_range(std::size_t count, std::size_t grain, const RangeFn& fn) const {
        if (run) {
            run(count, grain, fn);
        } else if (count > 0) {
            for (std::size_t begin = 0; begin < count; begin += grain) {
                fn(begin, std::min(count, begin + grain));
            }
        }
    }
};

[[nodiscard]] constexpr std::size_t range_count(std::size_t count, std::size_t grain) {
    return (count + grain - 1) / grain;
}

}
