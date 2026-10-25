#include "model/normalization_run.h"

#include "test_support.h"

#include <gtest/gtest.h>

namespace game {

namespace {

// A source folder with `per_species` cat photos (red) and as many dog photos (blue), plus one broken photo of each.
SourceScan make_source(const test::TempDir& dir, int per_species) {
    for (int i = 0; i < per_species; ++i) {
        test::write_bmp(dir / "cats" / (std::to_string(i) + ".bmp"), 40 + i, 30, test::Rgb{250, 0, 0});
        test::write_bmp(dir / "dogs" / (std::to_string(i) + ".bmp"), 30, 40 + i, test::Rgb{0, 0, 250});
    }
    test::write_text(dir / "cats" / "broken.jpg", "not a picture");
    test::write_text(dir / "dogs" / "empty.jpg", "");
    return scan_source_folder(dir.path()).value();
}

}

TEST(NormalizationRun, ConvertsEveryReadablePhotoAndSkipsTheRest) {
    const test::TempDir dir;
    NormalizationRun run{make_source(dir, 5), Parallel{}};
    EXPECT_EQ(run.total(), 12u);
    EXPECT_EQ(run.result().class_names, (ClassNames{"cats", "dogs"}));
    while (!run.done()) {
        run.step(4);
    }
    EXPECT_EQ(run.processed(), 12u);
    EXPECT_EQ(run.skipped(), 2u);
    EXPECT_EQ(run.failures().size(), 2u);
    ASSERT_EQ(run.result().size(), 10u);
    EXPECT_EQ(std::ranges::count(run.result().labels, static_cast<std::uint8_t>(ClassId::First)), 5);
    EXPECT_EQ(std::ranges::count(run.result().labels, static_cast<std::uint8_t>(ClassId::Second)), 5);
}

TEST(NormalizationRun, PicturesKeepTheirLabelsAndColors) {
    const test::TempDir dir;
    NormalizationRun run{make_source(dir, 3), Parallel{}};
    while (!run.done()) {
        run.step(100);
    }
    const LabeledImages data = run.take_result();
    for (std::size_t i = 0; i < data.size(); ++i) {
        const auto picture = data.image(i);
        const bool cat = data.labels[i] == static_cast<std::uint8_t>(ClassId::First);
        EXPECT_EQ(picture[0], cat ? 250 : 0);
        EXPECT_EQ(picture[2], cat ? 0 : 250);
    }
}

TEST(NormalizationRun, ThreadsGiveTheSameDatasetAsOneThread) {
    const test::TempDir dir;
    const auto scan = make_source(dir, 12);
    NormalizationRun serial{scan, Parallel{}};
    NormalizationRun parallel{scan, test::threaded()};
    while (!serial.done()) {
        serial.step(7);
    }
    while (!parallel.done()) {
        parallel.step(7);
    }
    EXPECT_EQ(serial.result().labels, parallel.result().labels);
    EXPECT_EQ(serial.result().images, parallel.result().images);
}

TEST(NormalizationRun, StepStopsAtTheEndAndKeepsCounting) {
    const test::TempDir dir;
    NormalizationRun run{make_source(dir, 1), Parallel{}};
    run.step(1);
    EXPECT_EQ(run.processed(), 1u);
    EXPECT_FALSE(run.done());
    run.step(1000);
    EXPECT_TRUE(run.done());
    run.step(1000);
    EXPECT_EQ(run.processed(), run.total());
}

TEST(NormalizationRun, ReportsAtMostTenFailures) {
    const test::TempDir dir;
    test::write_bmp(dir / "dogs" / "ok.bmp", 8, 8, test::Rgb{1, 2, 3});
    for (int i = 0; i < 25; ++i) {
        test::write_text(dir / "cats" / (std::to_string(i) + ".jpg"), "bad");
    }
    NormalizationRun run{scan_source_folder(dir.path()).value(), Parallel{}};
    run.step(1000);
    EXPECT_EQ(run.skipped(), 25u);
    EXPECT_EQ(run.failures().size(), NormalizationRun::kMaxReportedFailures);
}

}
