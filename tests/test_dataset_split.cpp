#include "model/dataset_split.h"

#include "test_support.h"

#include <gtest/gtest.h>

#include <algorithm>

namespace game {

namespace {

// `cats` cats then `dogs` dogs, in that order.
LabeledImages labeled(std::size_t cats, std::size_t dogs) {
    LabeledImages data;
    const std::vector<std::uint8_t> picture(kImageBytes, 0);
    for (std::size_t i = 0; i < cats; ++i) {
        data.add(picture, ClassId::First);
    }
    for (std::size_t i = 0; i < dogs; ++i) {
        data.add(picture, ClassId::Second);
    }
    return data;
}

std::size_t count_of(const LabeledImages& data, const std::vector<std::size_t>& part, ClassId species) {
    return static_cast<std::size_t>(std::ranges::count_if(part, [&](std::size_t index) {
        return data.labels[index] == static_cast<std::uint8_t>(species);
    }));
}

}

TEST(DatasetSplit, PartsAreDisjointAndCoverTheDataset) {
    const LabeledImages data = labeled(240, 500);
    const DatasetSplit split = split_dataset(data, 0.2, 1);
    std::vector<std::size_t> all = split.train;
    all.insert(all.end(), split.validation.begin(), split.validation.end());
    std::ranges::sort(all);
    ASSERT_EQ(all.size(), data.size());
    for (std::size_t i = 0; i < all.size(); ++i) {
        ASSERT_EQ(all[i], i);
    }
}

TEST(DatasetSplit, IndicesComeInAscendingOrder) {
    const DatasetSplit split = split_dataset(labeled(50, 80), 0.25, 7);
    EXPECT_TRUE(std::ranges::is_sorted(split.train));
    EXPECT_TRUE(std::ranges::is_sorted(split.validation));
}

TEST(DatasetSplit, KeepsTheShareOfEachSpeciesInBothParts) {
    const LabeledImages data = labeled(2394, 4990);
    const DatasetSplit split = split_dataset(data, 0.2, 42);
    EXPECT_EQ(count_of(data, split.validation, ClassId::First), 479u);   // 2394 * 0.2 = 478.8
    EXPECT_EQ(count_of(data, split.validation, ClassId::Second), 998u);   // 4990 * 0.2
    EXPECT_EQ(count_of(data, split.train, ClassId::First), 2394u - 479u);
    EXPECT_EQ(count_of(data, split.train, ClassId::Second), 4990u - 998u);
}

TEST(DatasetSplit, SameSeedGivesTheSameSplit) {
    const LabeledImages data = labeled(100, 100);
    const DatasetSplit a = split_dataset(data, 0.3, 5);
    const DatasetSplit b = split_dataset(data, 0.3, 5);
    EXPECT_EQ(a.train, b.train);
    EXPECT_EQ(a.validation, b.validation);
}

TEST(DatasetSplit, OtherSeedGivesAnotherSplit) {
    const LabeledImages data = labeled(100, 100);
    EXPECT_NE(split_dataset(data, 0.3, 5).validation, split_dataset(data, 0.3, 6).validation);
}

TEST(DatasetSplit, ValidationIsMixedNotTheFirstSamples) {
    // The dataset lists cats first; a split that cut the end off would put dogs only in validation.
    const LabeledImages data = labeled(100, 100);
    const DatasetSplit split = split_dataset(data, 0.2, 3);
    EXPECT_EQ(count_of(data, split.validation, ClassId::First), 20u);
    EXPECT_EQ(count_of(data, split.validation, ClassId::Second), 20u);
    EXPECT_LT(split.validation.front(), 100u);
    EXPECT_GE(split.validation.back(), 100u);
}

TEST(DatasetSplit, EdgeFractions) {
    const LabeledImages data = labeled(10, 10);
    EXPECT_TRUE(split_dataset(data, 0.0, 1).validation.empty());
    EXPECT_EQ(split_dataset(data, 0.0, 1).train.size(), 20u);
    EXPECT_TRUE(split_dataset(data, 1.0, 1).train.empty());
    EXPECT_EQ(split_dataset(data, 1.0, 1).validation.size(), 20u);
    EXPECT_EQ(split_dataset(data, -3.0, 1).validation.size(), 0u);
    EXPECT_EQ(split_dataset(data, 9.0, 1).train.size(), 0u);
}

TEST(DatasetSplit, NoSpeciesVanishesFromEitherPart) {
    // 20% of 3 rounds to 1 sample, and 1% of 3 rounds to 0: the species still keeps one in each part.
    const LabeledImages data = labeled(3, 400);
    for (const double fraction : {0.01, 0.2, 0.99}) {
        const DatasetSplit split = split_dataset(data, fraction, 1);
        EXPECT_GE(count_of(data, split.train, ClassId::First), 1u) << fraction;
        EXPECT_GE(count_of(data, split.validation, ClassId::First), 1u) << fraction;
        EXPECT_GE(count_of(data, split.train, ClassId::Second), 1u) << fraction;
        EXPECT_GE(count_of(data, split.validation, ClassId::Second), 1u) << fraction;
    }
}

TEST(DatasetSplit, EmptyAndSingleSpeciesDatasets) {
    EXPECT_TRUE(split_dataset({}, 0.2, 1).train.empty());
    EXPECT_TRUE(split_dataset({}, 0.2, 1).validation.empty());

    const DatasetSplit dogs = split_dataset(labeled(0, 10), 0.2, 1);
    EXPECT_EQ(dogs.validation.size(), 2u);
    EXPECT_EQ(dogs.train.size(), 8u);

    const DatasetSplit one = split_dataset(labeled(0, 1), 0.5, 1);
    EXPECT_EQ(one.train.size() + one.validation.size(), 1u);
}

}
