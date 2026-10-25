#include "model/normalization_run.h"

#include "model/image_decoder.h"
#include "model/image_resize.h"

#include <algorithm>

namespace game {

NormalizationRun::NormalizationRun(SourceScan scan, Parallel parallel)
    : files_(std::move(scan.files))
    , parallel_(std::move(parallel)) {
    result_.class_names = std::move(scan.class_names);
}

void NormalizationRun::step(std::size_t count) {
    const std::size_t first = next_;
    const std::size_t last = std::min(files_.size(), first + count);
    if (first >= last) {
        return;
    }
    const std::size_t batch = last - first;

    std::vector<std::uint8_t> pixels(batch * kImageBytes);
    std::vector<std::string> errors(batch);  // empty: the photo was read
    parallel_.for_each_range(batch, parallel_.grain_for(batch, 4), [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            const auto image = decode_image(files_[first + i].path);
            if (!image) {
                errors[i] = image.error();
                continue;
            }
            const auto picture = to_square_picture(*image);
            std::ranges::copy(picture, pixels.begin() + static_cast<std::ptrdiff_t>(i * kImageBytes));
        }
    });

    for (std::size_t i = 0; i < batch; ++i) {
        if (!errors[i].empty()) {
            ++skipped_;
            if (failures_.size() < kMaxReportedFailures) {
                failures_.push_back(std::move(errors[i]));
            }
            continue;
        }
        result_.add(std::span<const std::uint8_t>(pixels).subspan(i * kImageBytes, kImageBytes),
                files_[first + i].class_id);
    }
    next_ = last;
}

}
