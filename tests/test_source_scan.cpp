#include "model/source_scan.h"

#include "test_support.h"

#include <gtest/gtest.h>

#include <algorithm>

namespace game {

TEST(SourceScan, TheSubfoldersAreTheClassesInTheOrderOfTheirNamesWhateverTheCase) {
    const test::TempDir dir;
    test::write_text(dir / "Male" / "b.jpg", "x");
    test::write_text(dir / "Male" / "a.PNG", "x");
    test::write_text(dir / "female" / "d.jpeg", "x");
    test::write_text(dir / "female" / "e.bmp", "x");
    const auto scan = scan_source_folder(dir.path());
    ASSERT_TRUE(scan) << scan.error();
    EXPECT_EQ(scan->class_names[0], "female");
    EXPECT_EQ(scan->class_names[1], "Male");
    ASSERT_EQ(scan->files.size(), 4u);
    EXPECT_EQ(std::ranges::count(scan->files, ClassId::First, &SourceFile::class_id), 2);
    EXPECT_EQ(std::ranges::count(scan->files, ClassId::Second, &SourceFile::class_id), 2);
    for (const SourceFile& file : scan->files) {
        EXPECT_EQ(file.path.parent_path().filename().string(),
                scan->class_names[static_cast<std::size_t>(file.class_id)]);
    }
    EXPECT_TRUE(std::ranges::is_sorted(scan->files, {}, &SourceFile::path));
}

TEST(SourceScan, IgnoresOtherFilesAndFoldersWithoutPhotos) {
    const test::TempDir dir;
    test::write_text(dir / "cats" / "a.jpg", "x");
    test::write_text(dir / "cats" / "notes.txt", "x");
    test::write_text(dir / "cats" / "a.mat", "x");
    test::write_text(dir / "dogs" / "a.jpg", "x");
    test::write_text(dir / "docs" / "readme.txt", "x");
    test::write_text(dir / "loose.jpg", "x");
    const auto scan = scan_source_folder(dir.path());
    ASSERT_TRUE(scan) << scan.error();
    EXPECT_EQ(scan->files.size(), 2u);
}

TEST(SourceScan, FindsPhotosInNestedFolders) {
    const test::TempDir dir;
    test::write_text(dir / "cats" / "siamese" / "a.jpg", "x");
    test::write_text(dir / "dogs" / "pug" / "deep" / "a.jpg", "x");
    const auto scan = scan_source_folder(dir.path());
    ASSERT_TRUE(scan) << scan.error();
    EXPECT_EQ(scan->files.size(), 2u);
}

TEST(SourceScan, NeedsExactlyTwoSubfoldersWithPhotos) {
    const test::TempDir dir;
    EXPECT_FALSE(scan_source_folder(dir.path()));
    test::write_text(dir / "cats" / "a.jpg", "x");
    EXPECT_FALSE(scan_source_folder(dir.path()));
    test::write_text(dir / "dogs" / "readme.txt", "x");
    EXPECT_FALSE(scan_source_folder(dir.path()));
    test::write_text(dir / "dogs" / "a.jpg", "x");
    EXPECT_TRUE(scan_source_folder(dir.path()));
    test::write_text(dir / "birds" / "a.jpg", "x");
    EXPECT_FALSE(scan_source_folder(dir.path()));
}

TEST(SourceScan, FailsForAMissingFolder) {
    const test::TempDir dir;
    EXPECT_FALSE(scan_source_folder(dir / "nowhere"));
}

}
