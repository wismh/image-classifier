#pragma once

#include "model/labeled_images.h"
#include "model/parallel.h"
#include "model/source_scan.h"

#include <string>
#include <vector>

namespace game {

// Turns source photos into a dataset a few at a time, so the window stays responsive: every call of step() decodes and
// shrinks the next photos on the worker threads. Photos that cannot be read are skipped and counted.
class NormalizationRun {
public:
    static constexpr std::size_t kMaxReportedFailures = 10;

    NormalizationRun(SourceScan scan, Parallel parallel);

    // Handles up to `count` more photos.
    void step(std::size_t count);

    [[nodiscard]] bool done() const { return next_ >= files_.size(); }
    [[nodiscard]] std::size_t total() const { return files_.size(); }
    [[nodiscard]] std::size_t processed() const { return next_; }
    [[nodiscard]] std::size_t skipped() const { return skipped_; }
    // Messages of the first skipped photos.
    [[nodiscard]] const std::vector<std::string>& failures() const { return failures_; }
    [[nodiscard]] const LabeledImages& result() const { return result_; }
    // The dataset so far; the run is left empty.
    [[nodiscard]] LabeledImages take_result() { return std::move(result_); }

private:
    std::vector<SourceFile> files_;
    Parallel parallel_;
    std::size_t next_ = 0;
    std::size_t skipped_ = 0;
    std::vector<std::string> failures_;
    LabeledImages result_;
};

}
