#include "model/training_run.h"

#include "test_support.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>

namespace game {

namespace {

constexpr CnnConfig kSmall{2, 2, 2, 8};

// A cat is dark, a dog is bright (test::synthetic_set): any network can learn that.
std::shared_ptr<const LabeledImages> easy_data(std::size_t count) {
    return std::make_shared<const LabeledImages>(test::synthetic_set(count, 1));
}

TrainingConfig quick_config(unsigned epochs) {
    TrainingConfig config;
    config.epochs = epochs;
    config.batch_size = 8;
    config.learning_rate = 5e-3f;
    config.augment = false;
    config.seed = 3;
    return config;
}

void run_to_the_end(TrainingRun& run) {
    while (!run.step(std::chrono::milliseconds{50})) {
    }
}

}

TEST(TrainingRun, LearnsAnEasyProblem) {
    TrainingRun run{Cnn{kSmall, 1}, easy_data(80), quick_config(6), Parallel{}};
    run_to_the_end(run);
    ASSERT_EQ(run.history().size(), 6u);
    EXPECT_GE(run.history().back().validation_accuracy, 0.95f);
    EXPECT_GE(run.history().back().balanced_accuracy, 0.95f);
    EXPECT_LT(run.history().back().train_loss, run.history().front().train_loss);
    EXPECT_LT(run.history().back().validation_loss, 0.5f);
}

TEST(TrainingRun, TheSplitFollowsTheSplitSeedNotTheRunSeed) {
    // Training a network further needs the check samples of the first run, whatever else changes between the runs.
    TrainingConfig first = quick_config(1);
    first.split_seed = 11;
    first.seed = 100;
    TrainingConfig again = first;
    again.seed = 200;
    TrainingConfig other_split = first;
    other_split.split_seed = 12;
    TrainingRun a{Cnn{kSmall, 1}, easy_data(80), first, Parallel{}};
    TrainingRun b{Cnn{kSmall, 1}, easy_data(80), again, Parallel{}};
    TrainingRun c{Cnn{kSmall, 1}, easy_data(80), other_split, Parallel{}};
    EXPECT_EQ(a.split().validation, b.split().validation);
    EXPECT_EQ(a.split().train, b.split().train);
    EXPECT_NE(a.split().validation, c.split().validation);
}

TEST(TrainingRun, ATrainedNetworkCanBeTrainedFurtherAndKeepsImproving) {
    TrainingConfig config = quick_config(3);
    TrainingRun first{Cnn{kSmall, 1}, easy_data(80), config, Parallel{}};
    run_to_the_end(first);
    const float loss_after_first = first.history().back().validation_loss;
    const Cnn trained = first.take_model();

    // A second run starts from the weights of the first and never from random ones: its first epoch is already as good.
    config.seed = 77;
    TrainingRun second{Cnn{trained}, easy_data(80), config, Parallel{}};
    run_to_the_end(second);
    EXPECT_LT(second.history().front().validation_loss, 0.5f);
    EXPECT_LE(second.history().back().validation_loss, loss_after_first * 1.5f + 0.05f);
    EXPECT_EQ(second.split().validation, first.split().validation);
}

TEST(TrainingRun, SplitsTheDatasetOnceAndKeepsBothParts) {
    TrainingRun run{Cnn{kSmall, 1}, easy_data(80), quick_config(1), Parallel{}};
    EXPECT_EQ(run.train_size(), 64u);
    EXPECT_EQ(run.validation_size(), 16u);
}

TEST(TrainingRun, ReportsEpochNumbersAndSpeciesAccuracies) {
    TrainingRun run{Cnn{kSmall, 1}, easy_data(40), quick_config(3), Parallel{}};
    run_to_the_end(run);
    ASSERT_EQ(run.history().size(), 3u);
    for (unsigned i = 0; i < 3; ++i) {
        const EpochStats& stats = run.history()[i];
        EXPECT_EQ(stats.epoch, i + 1);
        EXPECT_NEAR(stats.balanced_accuracy, 0.5f * (stats.first_accuracy + stats.second_accuracy), 1e-6);
        EXPECT_GE(stats.train_accuracy, 0.0f);
        EXPECT_LE(stats.train_accuracy, 1.0f);
    }
    EXPECT_EQ(run.completed_epochs(), 3u);
    EXPECT_EQ(run.phase(), TrainingRun::Phase::Done);
    EXPECT_FLOAT_EQ(run.epoch_progress(), 1.0f);
}

TEST(TrainingRun, SameSeedGivesTheSameRun) {
    TrainingRun a{Cnn{kSmall, 1}, easy_data(40), quick_config(2), Parallel{}};
    TrainingRun b{Cnn{kSmall, 1}, easy_data(40), quick_config(2), Parallel{}};
    run_to_the_end(a);
    run_to_the_end(b);
    ASSERT_EQ(a.history().size(), b.history().size());
    for (std::size_t i = 0; i < a.history().size(); ++i) {
        EXPECT_EQ(a.history()[i].train_loss, b.history()[i].train_loss);
        EXPECT_EQ(a.history()[i].validation_loss, b.history()[i].validation_loss);
    }
    const Cnn model_a = a.take_model();
    const Cnn model_b = b.take_model();
    EXPECT_TRUE(std::ranges::equal(model_a.params(), model_b.params()));
}

TEST(TrainingRun, ThreadsGiveTheSameResultAsOneThread) {
    TrainingConfig config = quick_config(2);
    config.augment = true;  // the changes depend on the sample, not on the thread that handles it
    TrainingRun serial{Cnn{kSmall, 1}, easy_data(40), config, Parallel{}};
    TrainingRun threaded{Cnn{kSmall, 1}, easy_data(40), config, test::threaded()};
    run_to_the_end(serial);
    run_to_the_end(threaded);
    for (std::size_t i = 0; i < serial.history().size(); ++i) {
        EXPECT_EQ(serial.history()[i].train_loss, threaded.history()[i].train_loss);
        EXPECT_EQ(serial.history()[i].validation_accuracy, threaded.history()[i].validation_accuracy);
    }
    EXPECT_TRUE(std::ranges::equal(serial.take_model().params(), threaded.take_model().params()));
}

TEST(TrainingRun, AugmentationChangesTheRun) {
    TrainingConfig plain = quick_config(1);
    TrainingConfig augmented = plain;
    augmented.augment = true;
    TrainingRun a{Cnn{kSmall, 1}, easy_data(40), plain, Parallel{}};
    TrainingRun b{Cnn{kSmall, 1}, easy_data(40), augmented, Parallel{}};
    run_to_the_end(a);
    run_to_the_end(b);
    EXPECT_NE(a.history().front().train_loss, b.history().front().train_loss);
}

TEST(TrainingRun, AZeroBudgetStillDoesOneUnitOfWorkAndProgressGrows) {
    TrainingRun run{Cnn{kSmall, 1}, easy_data(80), quick_config(2), Parallel{}};
    EXPECT_FLOAT_EQ(run.epoch_progress(), 0.0f);
    float last = 0.0f;
    unsigned calls = 0;
    unsigned epochs_seen = 0;
    while (!run.step(std::chrono::steady_clock::duration::zero())) {
        ++calls;
        const unsigned epochs_now = run.completed_epochs();
        if (epochs_now != epochs_seen) {
            epochs_seen = epochs_now;
            last = 0.0f;  // a new epoch starts again from the beginning
        }
        EXPECT_GE(run.epoch_progress(), last);
        last = run.epoch_progress();
    }
    // 64 training samples in batches of 8 and 16 validation samples in one chunk: 9 units an epoch.
    EXPECT_EQ(calls, 2u * 9u - 1u);
}

TEST(TrainingRun, GoesThroughTheValidationPhaseAfterTraining) {
    TrainingRun run{Cnn{kSmall, 1}, easy_data(80), quick_config(1), Parallel{}};
    unsigned batches = 0;
    while (run.phase() == TrainingRun::Phase::Training) {
        run.step(std::chrono::steady_clock::duration::zero());
        ++batches;
    }
    EXPECT_EQ(batches, 8u);
    EXPECT_EQ(run.phase(), TrainingRun::Phase::Validating);
    EXPECT_NEAR(run.epoch_progress(), 64.0f / 80.0f, 1e-6);
    EXPECT_TRUE(run.step(std::chrono::steady_clock::duration::zero()));
}

TEST(TrainingRun, WorksWithoutAValidationPart) {
    TrainingConfig config = quick_config(2);
    config.validation_fraction = 0.0;
    TrainingRun run{Cnn{kSmall, 1}, easy_data(40), config, Parallel{}};
    EXPECT_EQ(run.validation_size(), 0u);
    run_to_the_end(run);
    ASSERT_EQ(run.history().size(), 2u);
    EXPECT_FLOAT_EQ(run.history().back().balanced_accuracy, run.history().back().train_accuracy);
}

TEST(TrainingRun, NothingToDoIsDoneAtOnce) {
    TrainingRun no_epochs{Cnn{kSmall, 1}, easy_data(40), quick_config(0), Parallel{}};
    EXPECT_TRUE(no_epochs.step(std::chrono::steady_clock::duration::zero()));
    EXPECT_TRUE(no_epochs.history().empty());

    TrainingRun no_data{Cnn{kSmall, 1}, std::make_shared<const LabeledImages>(), quick_config(3), Parallel{}};
    EXPECT_TRUE(no_data.step(std::chrono::steady_clock::duration::zero()));
    EXPECT_TRUE(no_data.history().empty());
}

TEST(TrainingRun, ReturnsTheNetworkOfTheEpochWithTheLowestCheckLoss) {
    TrainingConfig config = quick_config(5);
    config.learning_rate = 0.05f;  // rough enough that epochs differ
    TrainingRun full{Cnn{kSmall, 1}, easy_data(80), config, Parallel{}};
    run_to_the_end(full);
    const auto& history = full.history();
    const auto best = std::ranges::min_element(history, {}, &EpochStats::validation_loss);
    EXPECT_EQ(full.best_epoch(), best->epoch);  // min_element: the first epoch with the lowest loss

    // A run stopped right after that epoch is the same run so far, so it ends with the same network.
    config.epochs = full.best_epoch();
    TrainingRun stopped{Cnn{kSmall, 1}, easy_data(80), config, Parallel{}};
    run_to_the_end(stopped);
    EXPECT_EQ(stopped.best_epoch(), full.best_epoch());
    EXPECT_TRUE(std::ranges::equal(full.take_model().params(), stopped.take_model().params()));
}

TEST(TrainingRun, WithoutAValidationPartTheTrainingLossPicksTheBestEpoch) {
    TrainingConfig config = quick_config(4);
    config.validation_fraction = 0.0;
    TrainingRun run{Cnn{kSmall, 1}, easy_data(40), config, Parallel{}};
    run_to_the_end(run);
    const auto best = std::ranges::min_element(run.history(), {}, &EpochStats::train_loss);
    EXPECT_EQ(run.best_epoch(), best->epoch);
}

TEST(LearningRateSchedule, StaysConstantWithoutDecay) {
    for (const double progress : {0.0, 0.3, 1.0}) {
        EXPECT_FLOAT_EQ(scheduled_learning_rate(2e-3f, false, progress), 2e-3f);
    }
}

TEST(LearningRateSchedule, FollowsAHalfCosineDownToAHundredth) {
    EXPECT_NEAR(scheduled_learning_rate(1.0f, true, 0.0), 1.0f, 1e-6);
    EXPECT_NEAR(scheduled_learning_rate(1.0f, true, 0.5), 0.505f, 1e-4);
    EXPECT_NEAR(scheduled_learning_rate(1.0f, true, 1.0), 0.01f, 1e-6);
    float last = 2.0f;
    for (int i = 0; i <= 100; ++i) {
        const float rate = scheduled_learning_rate(1.0f, true, i / 100.0);
        EXPECT_LE(rate, last);
        last = rate;
    }
    // Outside the run it stays at the ends.
    EXPECT_NEAR(scheduled_learning_rate(1.0f, true, -1.0), 1.0f, 1e-6);
    EXPECT_NEAR(scheduled_learning_rate(1.0f, true, 7.0), 0.01f, 1e-6);
}

TEST(TrainingRun, DecayingTheLearningRateChangesTheRunAndKeepsItReproducible) {
    TrainingConfig steady = quick_config(3);
    TrainingConfig decaying = steady;
    decaying.lr_decay = true;
    TrainingRun a{Cnn{kSmall, 1}, easy_data(40), steady, Parallel{}};
    TrainingRun b{Cnn{kSmall, 1}, easy_data(40), decaying, Parallel{}};
    TrainingRun c{Cnn{kSmall, 1}, easy_data(40), decaying, Parallel{}};
    run_to_the_end(a);
    run_to_the_end(b);
    run_to_the_end(c);
    EXPECT_NE(a.history().back().train_loss, b.history().back().train_loss);
    EXPECT_EQ(b.history().back().train_loss, c.history().back().train_loss);
}

}
