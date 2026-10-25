#include "model/training_run.h"

#include "kernels/vector_ops.h"
#include "model/loss.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>

namespace game {

float scheduled_learning_rate(float base, bool decay, double progress) {
    if (!decay) {
        return base;
    }
    constexpr double kFloor = 0.01;
    const double t = std::clamp(progress, 0.0, 1.0);
    return static_cast<float>(base * (kFloor + (1.0 - kFloor) * 0.5 * (1.0 + std::cos(std::numbers::pi * t))));
}

void TrainingRun::Worker::reset(bool with_grads) {
    if (with_grads) {
        std::ranges::fill(grads, 0.0f);
    }
    loss = 0.0;
    std::ranges::fill(seen, 0);
    std::ranges::fill(correct, 0);
}

TrainingRun::TrainingRun(Cnn model, std::shared_ptr<const LabeledImages> data, const TrainingConfig& config,
        Parallel parallel)
    : model_(std::move(model))
    , data_(std::move(data))
    , config_(config)
    , parallel_(std::move(parallel))
    , split_(split_dataset(*data_, config.validation_fraction, config.split_seed))
    , adam_(model_.param_count())
    , batch_grads_(model_.param_count(), 0.0f) {
    std::size_t per_class[kNumClasses]{};
    for (const std::size_t index : split_.train) {
        ++per_class[data_->labels[index]];
    }
    for (std::size_t c = 0; c < kNumClasses; ++c) {
        // Each class counts as 1 / (classes * its share), so the weights of all samples add up to the sample count.
        class_weight_[c] = per_class[c] == 0 ? 1.0f
                : static_cast<float>(split_.train.size()) / (static_cast<float>(kNumClasses) * static_cast<float>(per_class[c]));
    }
    config_.batch_size = std::max<std::size_t>(1, config_.batch_size);
    if (split_.train.empty() || config_.epochs == 0) {
        phase_ = Phase::Done;
        return;
    }
    batches_total_ = static_cast<std::size_t>(config_.epochs)
            * ((split_.train.size() + config_.batch_size - 1) / config_.batch_size);
    start_epoch();
}

std::vector<TrainingRun::Worker>& TrainingRun::workers_for(std::size_t samples) {
    const std::size_t ranges = range_count(samples, kSamplesPerRange);
    while (workers_.size() < ranges) {
        Worker worker;
        worker.workspace = model_.make_workspace();
        worker.input.resize(Cnn::kInputSize);
        worker.d_scores.resize(kNumClasses);
        worker.grads.assign(model_.param_count(), 0.0f);
        workers_.push_back(std::move(worker));
    }
    return workers_;
}

float TrainingRun::epoch_progress() const {
    const std::size_t total = split_.train.size() + split_.validation.size();
    if (total == 0 || phase_ == Phase::Done) {
        return phase_ == Phase::Done ? 1.0f : 0.0f;
    }
    const std::size_t done = phase_ == Phase::Training ? cursor_ : split_.train.size() + validation_cursor_;
    return static_cast<float>(done) / static_cast<float>(total);
}

bool TrainingRun::step(std::chrono::steady_clock::duration budget) {
    const auto start = std::chrono::steady_clock::now();
    do {
        switch (phase_) {
        case Phase::Training: run_batch(); break;
        case Phase::Validating: run_validation_chunk(); break;
        case Phase::Done: return true;
        }
    } while (phase_ != Phase::Done && std::chrono::steady_clock::now() - start < budget);
    return phase_ == Phase::Done;
}

void TrainingRun::start_epoch() {
    const unsigned epoch = completed_epochs();
    order_ = split_.train;
    std::mt19937 rng = sample_rng(config_.seed, epoch, split_.train.size());
    for (std::size_t i = order_.size(); i > 1; --i) {
        std::swap(order_[i - 1], order_[rng() % i]);
    }
    cursor_ = 0;
    validation_cursor_ = 0;
    train_loss_sum_ = 0.0;
    train_seen_ = 0;
    train_correct_ = 0;
    validation_loss_sum_ = 0.0;
    std::ranges::fill(validation_seen_, 0);
    std::ranges::fill(validation_correct_, 0);
    phase_ = Phase::Training;
}

void TrainingRun::run_batch() {
    const std::size_t count = std::min(config_.batch_size, order_.size() - cursor_);
    const std::size_t first = cursor_;
    const unsigned epoch = completed_epochs();

    std::vector<Worker>& workers = workers_for(count);
    const std::size_t ranges = range_count(count, kSamplesPerRange);
    for (std::size_t r = 0; r < ranges; ++r) {
        workers[r].reset(false);
    }

    parallel_.for_each_range(count, kSamplesPerRange, [&](std::size_t begin, std::size_t end) {
        Worker& worker = workers[begin / kSamplesPerRange];
        std::ranges::fill(worker.grads, 0.0f);  // in the range's own thread: the gradients are the biggest buffers
        for (std::size_t i = begin; i < end; ++i) {
            const std::size_t sample = order_[first + i];
            const std::span<const std::uint8_t> picture = data_->image(sample);
            if (config_.augment) {
                std::mt19937 rng = sample_rng(config_.seed, epoch, sample);
                to_augmented_input(picture, worker.input, config_.augmentation, rng);
            } else {
                to_input(picture, worker.input);
            }
            const int label = data_->labels[sample];
            const std::span<const float> scores = model_.forward(worker.input, worker.workspace);
            const LossResult result = softmax_cross_entropy(scores, label, class_weight_[label], worker.d_scores);
            model_.backward(worker.input, worker.workspace, worker.d_scores, worker.grads);
            worker.loss += result.loss;
            ++worker.seen[label];
            worker.correct[label] += result.predicted == label ? 1 : 0;
        }
    });

    for (std::size_t r = 0; r < ranges; ++r) {
        train_loss_sum_ += workers[r].loss;
        for (std::size_t c = 0; c < kNumClasses; ++c) {
            train_seen_ += workers[r].seen[c];
            train_correct_ += workers[r].correct[c];
        }
    }

    // The gradients of the ranges are added up, and the step is taken, piece by piece of the parameter vector on all
    // the
    // threads. Each parameter's gradient is added in range order, so the sum does not depend on which thread finished
    // first.
    const double progress = static_cast<double>(batches_done_) / static_cast<double>(std::max<std::size_t>(1, batches_total_));
    const float learning_rate = scheduled_learning_rate(config_.learning_rate, config_.lr_decay, progress);
    const float scale = 1.0f / static_cast<float>(count);
    adam_.begin_step();
    parallel_.for_each_range(batch_grads_.size(), kParamsPerPiece, [&](std::size_t begin, std::size_t end) {
        std::fill(batch_grads_.begin() + static_cast<std::ptrdiff_t>(begin),
                batch_grads_.begin() + static_cast<std::ptrdiff_t>(end), 0.0f);
        for (std::size_t r = 0; r < ranges; ++r) {
            add_floats(batch_grads_.data() + begin, workers[r].grads.data() + begin, end - begin);
        }
        adam_.update_range(model_.params(), batch_grads_, begin, end, learning_rate, scale);
    });
    ++batches_done_;

    cursor_ += count;
    if (cursor_ >= order_.size()) {
        finish_training();
    }
}

void TrainingRun::finish_training() {
    if (split_.validation.empty()) {
        finish_epoch();
    } else {
        phase_ = Phase::Validating;
        validation_cursor_ = 0;
    }
}

void TrainingRun::run_validation_chunk() {
    const std::size_t count = std::min(kValidationChunk, split_.validation.size() - validation_cursor_);
    const std::size_t first = validation_cursor_;

    std::vector<Worker>& workers = workers_for(count);
    const std::size_t ranges = range_count(count, kSamplesPerRange);
    for (std::size_t r = 0; r < ranges; ++r) {
        workers[r].reset(false);
    }

    parallel_.for_each_range(count, kSamplesPerRange, [&](std::size_t begin, std::size_t end) {
        Worker& worker = workers[begin / kSamplesPerRange];
        for (std::size_t i = begin; i < end; ++i) {
            const std::size_t sample = split_.validation[first + i];
            to_input(data_->image(sample), worker.input);
            const int label = data_->labels[sample];
            const std::span<const float> scores = model_.forward(worker.input, worker.workspace);
            const LossResult result = softmax_cross_entropy(scores, label, 1.0f, worker.d_scores);
            worker.loss += result.loss;
            ++worker.seen[label];
            worker.correct[label] += result.predicted == label ? 1 : 0;
        }
    });

    for (std::size_t r = 0; r < ranges; ++r) {
        validation_loss_sum_ += workers[r].loss;
        for (std::size_t c = 0; c < kNumClasses; ++c) {
            validation_seen_[c] += workers[r].seen[c];
            validation_correct_[c] += workers[r].correct[c];
        }
    }
    validation_cursor_ += count;
    if (validation_cursor_ >= split_.validation.size()) {
        finish_epoch();
    }
}

void TrainingRun::finish_epoch() {
    const auto share = [](std::size_t part, std::size_t whole) {
        return whole == 0 ? 0.0f : static_cast<float>(part) / static_cast<float>(whole);
    };
    EpochStats stats;
    stats.epoch = completed_epochs() + 1;
    stats.train_loss = train_seen_ == 0 ? 0.0f : static_cast<float>(train_loss_sum_ / static_cast<double>(train_seen_));
    stats.train_accuracy = share(train_correct_, train_seen_);

    const std::size_t validated = validation_seen_[0] + validation_seen_[1];
    stats.validation_loss = validated == 0 ? 0.0f : static_cast<float>(validation_loss_sum_ / static_cast<double>(validated));
    stats.validation_accuracy = share(validation_correct_[0] + validation_correct_[1], validated);
    stats.first_accuracy = share(validation_correct_[0], validation_seen_[0]);
    stats.second_accuracy = share(validation_correct_[1], validation_seen_[1]);
    // Without a validation part, the training numbers stand in for it.
    stats.balanced_accuracy = validated == 0 ? stats.train_accuracy : 0.5f * (stats.first_accuracy + stats.second_accuracy);
    history_.push_back(stats);

    // Without a validation part, the training loss stands in for it.
    const float loss = validated == 0 ? stats.train_loss : stats.validation_loss;
    if (best_epoch_ == 0 || loss < best_loss_) {
        best_loss_ = loss;
        best_epoch_ = stats.epoch;
        best_params_.assign(model_.params().begin(), model_.params().end());
    }

    if (completed_epochs() >= config_.epochs) {
        phase_ = Phase::Done;
    } else {
        start_epoch();
    }
}

Cnn TrainingRun::take_model() {
    if (!best_params_.empty()) {
        std::ranges::copy(best_params_, model_.params().begin());
    }
    return std::move(model_);
}

}
