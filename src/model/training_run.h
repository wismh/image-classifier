#pragma once

#include "model/adam.h"
#include "model/augment.h"
#include "model/cnn.h"
#include "model/dataset_split.h"
#include "model/labeled_images.h"
#include "model/parallel.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

namespace game {

struct TrainingConfig {
    unsigned epochs = 10;
    std::size_t batch_size = 32;
    float learning_rate = 1e-3f;
    // Lower the learning rate over the run along a half cosine, from `learning_rate` down to a hundredth of it, so the
    // last epochs make small corrections instead of the large steps that make the check numbers jump from epoch to
    // epoch.
    bool lr_decay = false;
    double validation_fraction = 0.2;
    bool augment = true;
    Augmentation augmentation;
    // The same seed gives the same order of samples and the same changes to them.
    std::uint32_t seed = 1;
    // The same split seed gives the same split of the dataset into training and validation samples. Kept apart from
    // `seed` so that a network trained further is checked on the samples it was checked on before.
    std::uint32_t split_seed = 1;
};

// The learning rate for a point `progress` (0..1) of the way through the run: `base` while `decay` is off, otherwise
// following a half cosine down to a hundredth of `base`.
[[nodiscard]] float scheduled_learning_rate(float base, bool decay, double progress);

// What one epoch came to. Train numbers are measured while training on the samples (so they include the effect of
// augmentation and of the weights moving); validation numbers are measured on the held-out samples after the epoch.
// Accuracy is a share 0..1. `balanced_accuracy` is the mean of the two species' accuracies, so a network that says
// "second class" to everything does not look good just because it has more samples.
struct EpochStats {
    unsigned epoch = 0;  // 1-based
    float train_loss = 0.0f;
    float train_accuracy = 0.0f;
    float validation_loss = 0.0f;
    float validation_accuracy = 0.0f;
    float first_accuracy = 0.0f;
    float second_accuracy = 0.0f;
    float balanced_accuracy = 0.0f;
};

// Mini-batch training of a Cnn with Adam, advanced in slices so the caller keeps its frame rate: each `step` works for
// a
// time budget and returns. The dataset is split once at the start (split_dataset); every epoch goes through the
// training part in a new random order, then through the validation part. Each batch is spread over `parallel`.
// Classes are weighted by the inverse of their share of the training part, so each class counts as much as the other.
class TrainingRun {
public:
    enum class Phase { Training, Validating, Done };

    // Validation passes this many samples per unit of work.
    static constexpr std::size_t kValidationChunk = 64;
    // The samples of a batch are cut into ranges of this many, whatever the number of threads, and the ranges' sums are
    // added in order: a run gives the same numbers on a machine with 2 cores and one with 16.
    static constexpr std::size_t kSamplesPerRange = 2;
    // The parameter vector is cut into pieces of this many for adding up gradients and for the step of Adam.
    static constexpr std::size_t kParamsPerPiece = 16384;

    TrainingRun(Cnn model, std::shared_ptr<const LabeledImages> data, const TrainingConfig& config, Parallel parallel);

    // Works for about `budget` (at least one unit of work). Returns true once every epoch is done.
    bool step(std::chrono::steady_clock::duration budget);

    [[nodiscard]] Phase phase() const { return phase_; }
    [[nodiscard]] unsigned epochs() const { return config_.epochs; }
    [[nodiscard]] unsigned completed_epochs() const { return static_cast<unsigned>(history_.size()); }
    [[nodiscard]] const std::vector<EpochStats>& history() const { return history_; }
    [[nodiscard]] const DatasetSplit& split() const { return split_; }
    [[nodiscard]] std::size_t train_size() const { return split_.train.size(); }
    [[nodiscard]] std::size_t validation_size() const { return split_.validation.size(); }
    // Share of the current epoch done (training and validation together), 0..1.
    [[nodiscard]] float epoch_progress() const;
    // The 1-based epoch with the lowest validation loss so far (the lowest training loss when there is no validation
    // part); 0 before the first finishes. Loss moves more smoothly than accuracy, which jumps with every sample near
    // the
    // border between the classes.
    [[nodiscard]] unsigned best_epoch() const { return best_epoch_; }

    // The network as it was after its best epoch (as it is when no epoch finished).
    [[nodiscard]] Cnn take_model();

private:
    // What one range of samples of a batch leaves behind; one per range so no two threads share any of it.
    struct Worker {
        Cnn::Workspace workspace;
        std::vector<float> input;
        std::vector<float> d_scores;
        std::vector<float> grads;
        double loss = 0.0;
        std::size_t seen[kNumClasses]{};
        std::size_t correct[kNumClasses]{};

        void reset(bool with_grads);
    };

    void run_batch();
    void run_validation_chunk();
    void start_epoch();
    void finish_training();
    void finish_epoch();
    // At least as many workers as `samples` need.
    [[nodiscard]] std::vector<Worker>& workers_for(std::size_t samples);

    Cnn model_;
    std::shared_ptr<const LabeledImages> data_;
    TrainingConfig config_;
    Parallel parallel_;
    DatasetSplit split_;
    Adam adam_;
    float class_weight_[kNumClasses]{};

    Phase phase_ = Phase::Training;
    std::vector<std::size_t> order_;  // the training samples in this epoch's order
    std::size_t cursor_ = 0;
    std::vector<Worker> workers_;
    std::vector<float> batch_grads_;

    // Sums over the current epoch.
    double train_loss_sum_ = 0.0;
    std::size_t train_seen_ = 0;
    std::size_t train_correct_ = 0;
    double validation_loss_sum_ = 0.0;
    std::size_t validation_seen_[kNumClasses]{};
    std::size_t validation_correct_[kNumClasses]{};
    std::size_t validation_cursor_ = 0;

    std::vector<EpochStats> history_;
    unsigned best_epoch_ = 0;
    float best_loss_ = 0.0f;
    std::size_t batches_done_ = 0;
    std::size_t batches_total_ = 0;
    std::vector<float> best_params_;
};

}
