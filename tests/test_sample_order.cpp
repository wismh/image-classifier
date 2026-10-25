#include "model/sample_order.h"

#include <gtest/gtest.h>

namespace game {

namespace {

LabeledImages labeled(const std::vector<ClassId>& classes) {
    LabeledImages data;
    const std::vector<std::uint8_t> pixels(kImageBytes, 0);
    for (const ClassId class_id : classes) {
        data.add(pixels, class_id);
    }
    return data;
}

}

TEST(SampleOrder, ListsTheFirstClassThenTheSecondEachInItsOwnOrder) {
    const SampleOrder order{
            labeled({ClassId::Second, ClassId::First, ClassId::Second, ClassId::First, ClassId::First})};
    ASSERT_EQ(order.size(), 5u);
    EXPECT_EQ(order.first_count(), 3u);
    EXPECT_EQ(order.second_count(), 2u);
    const std::vector<std::size_t> expected{1, 3, 4, 0, 2};
    for (std::size_t position = 0; position < expected.size(); ++position) {
        EXPECT_EQ(order.sample_at(position), expected[position]);
    }
}

TEST(SampleOrder, PositionOfIsTheInverseOfSampleAt) {
    const SampleOrder order{labeled({ClassId::Second, ClassId::First, ClassId::Second, ClassId::First})};
    for (std::size_t position = 0; position < order.size(); ++position) {
        EXPECT_EQ(order.position_of(order.sample_at(position)), position);
    }
}

TEST(SampleOrder, AnEmptyDatasetHasOneEmptyPage) {
    const SampleOrder order{LabeledImages{}};
    EXPECT_EQ(order.size(), 0u);
    EXPECT_EQ(order.page_count(10), 1u);
}

TEST(SampleOrder, PagesRoundUp) {
    const SampleOrder order{labeled(std::vector<ClassId>(25, ClassId::First))};
    EXPECT_EQ(order.page_count(10), 3u);
    EXPECT_EQ(order.page_count(25), 1u);
    EXPECT_EQ(order.page_count(24), 2u);
}

TEST(LabeledImages, CountsTheSamplesOfAClass) {
    const LabeledImages data = labeled({ClassId::First, ClassId::Second, ClassId::Second});
    EXPECT_EQ(data.count(ClassId::First), 1u);
    EXPECT_EQ(data.count(ClassId::Second), 2u);
}

}
