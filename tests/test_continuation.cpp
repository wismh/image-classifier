#include "model/continuation.h"

#include <gtest/gtest.h>

namespace game {

namespace {

LabeledImages dataset(std::size_t samples, ClassNames names) {
    LabeledImages data;
    data.class_names = std::move(names);
    const std::vector<std::uint8_t> pixels(kImageBytes, 0);
    for (std::size_t i = 0; i < samples; ++i) {
        data.add(pixels, i % 2 == 0 ? ClassId::First : ClassId::Second);
    }
    return data;
}

ModelFile saved(std::size_t samples, ClassNames names, bool split_known) {
    ModelFile file;
    file.class_names = std::move(names);
    file.split_known = split_known;
    file.split_seed = 77;
    file.dataset_samples = split_known ? static_cast<unsigned>(samples) : 0;
    return file;
}

}

TEST(PlanContinuation, TheSameDataKeepsTheRecordedSplit) {
    const auto plan = plan_continuation(saved(20, {"cat", "dog"}, true), dataset(20, {"cat", "dog"}), 5);
    EXPECT_EQ(plan.kind, ContinueKind::SameData);
    EXPECT_EQ(plan.split_seed, 77u);
}

TEST(PlanContinuation, ADatasetOfAnotherSizeStillKeepsTheRecordedSplit) {
    const auto plan = plan_continuation(saved(20, {"cat", "dog"}, true), dataset(30, {"cat", "dog"}), 5);
    EXPECT_EQ(plan.kind, ContinueKind::OtherDataset);
    EXPECT_EQ(plan.split_seed, 77u);
}

TEST(PlanContinuation, AnOldFileHasNoSplitToKeepSoTheFreshSeedIsUsed) {
    const auto plan = plan_continuation(saved(20, {"cat", "dog"}, false), dataset(20, {"cat", "dog"}), 5);
    EXPECT_EQ(plan.kind, ContinueKind::UnknownSplit);
    EXPECT_EQ(plan.split_seed, 5u);
}

TEST(PlanContinuation, OtherClassesCannotBeContinued) {
    const auto plan = plan_continuation(saved(20, {"cat", "dog"}, true), dataset(20, {"male", "female"}), 5);
    EXPECT_EQ(plan.kind, ContinueKind::OtherClasses);
}

}
