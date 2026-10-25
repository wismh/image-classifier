#include "model/sample_order.h"

#include <algorithm>

namespace game {

SampleOrder::SampleOrder(const LabeledImages& data)
    : order_(data.size())
    , position_(data.size())
    , first_count_(data.count(ClassId::First)) {
    std::size_t next_first = 0;
    std::size_t next_second = first_count_;
    for (std::size_t sample = 0; sample < data.size(); ++sample) {
        const bool first = data.labels[sample] == static_cast<std::uint8_t>(ClassId::First);
        const std::size_t position = first ? next_first++ : next_second++;
        order_[position] = sample;
        position_[sample] = position;
    }
}

std::size_t SampleOrder::page_count(std::size_t page_samples) const {
    return std::max<std::size_t>(1, (order_.size() + page_samples - 1) / page_samples);
}

}
