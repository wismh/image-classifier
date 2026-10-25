#pragma once

#include "model/labeled_images.h"

#include <cstddef>
#include <vector>

namespace game {

// The samples of a dataset listed by class: the first class first, then the second, each in the order it came. A grid
// shows this list a page of samples at a time.
class SampleOrder {
public:
    explicit SampleOrder(const LabeledImages& data);

    [[nodiscard]] std::size_t size() const { return order_.size(); }
    [[nodiscard]] std::size_t first_count() const { return first_count_; }
    [[nodiscard]] std::size_t second_count() const { return order_.size() - first_count_; }
    // The index of the sample (in the dataset) shown at `position`.
    [[nodiscard]] std::size_t sample_at(std::size_t position) const { return order_[position]; }
    // Where the sample with dataset index `sample` is shown.
    [[nodiscard]] std::size_t position_of(std::size_t sample) const { return position_[sample]; }

    // Pages needed for the list; an empty list still has one (empty) page.
    [[nodiscard]] std::size_t page_count(std::size_t page_samples) const;

private:
    std::vector<std::size_t> order_;     // position -> sample
    std::vector<std::size_t> position_;  // sample -> position
    std::size_t first_count_ = 0;
};

}
