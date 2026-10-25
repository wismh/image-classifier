#include "model/dataset_file.h"

#include "test_support.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

namespace game {

TEST(DatasetFile, RoundTripKeepsLabelsAndPixels) {
    const test::TempDir dir;
    const LabeledImages data = test::synthetic_set(7, 1);
    ASSERT_TRUE(write_dataset_file(dir / "pets.cvds", data));

    const auto loaded = read_dataset_file(dir / "pets.cvds");
    ASSERT_TRUE(loaded) << loaded.error();
    EXPECT_EQ(loaded->labels, data.labels);
    EXPECT_EQ(loaded->images, data.images);
    EXPECT_FALSE(std::filesystem::exists(dir / "pets.cvds.tmp"));
}

TEST(DatasetFile, RoundTripKeepsTheNamesOfTheClasses) {
    const test::TempDir dir;
    LabeledImages data = test::synthetic_set(3, 1);
    data.class_names = {"female", "male"};
    ASSERT_TRUE(write_dataset_file(dir / "people.cvds", data));
    const auto loaded = read_dataset_file(dir / "people.cvds");
    ASSERT_TRUE(loaded) << loaded.error();
    EXPECT_EQ(loaded->class_names, (ClassNames{"female", "male"}));
}

TEST(DatasetFile, ReadsAVersion1FileAsCatsAndDogs) {
    // Written by hand the way version 1 of the program wrote it: no names after the number of samples.
    const test::TempDir dir;
    const LabeledImages data = test::synthetic_set(3, 1);
    std::vector<std::uint8_t> bytes{'C', 'V', 'D', 'S'};
    test::append(bytes, std::uint32_t{1});
    test::append(bytes, static_cast<std::uint32_t>(kImageSide));
    test::append(bytes, static_cast<std::uint32_t>(kChannels));
    test::append(bytes, static_cast<std::uint32_t>(data.size()));
    bytes.insert(bytes.end(), data.labels.begin(), data.labels.end());
    bytes.insert(bytes.end(), data.images.begin(), data.images.end());
    test::write_bytes(dir / "old.cvds", bytes);

    const auto loaded = read_dataset_file(dir / "old.cvds");
    ASSERT_TRUE(loaded) << loaded.error();
    EXPECT_EQ(loaded->class_names, (ClassNames{"cat", "dog"}));
    EXPECT_EQ(loaded->labels, data.labels);
    EXPECT_EQ(loaded->images, data.images);
}

TEST(DatasetFile, WriteCreatesMissingFolders) {
    const test::TempDir dir;
    ASSERT_TRUE(write_dataset_file(dir / "a" / "b" / "pets.cvds", test::synthetic_set(2, 1)));
    EXPECT_TRUE(std::filesystem::exists(dir / "a" / "b" / "pets.cvds"));
}

TEST(DatasetFile, EmptyDatasetRoundTrips) {
    const test::TempDir dir;
    ASSERT_TRUE(write_dataset_file(dir / "empty.cvds", {}));
    const auto loaded = read_dataset_file(dir / "empty.cvds");
    ASSERT_TRUE(loaded) << loaded.error();
    EXPECT_EQ(loaded->size(), 0u);
}

TEST(DatasetFile, RejectsMissingAndForeignFiles) {
    const test::TempDir dir;
    EXPECT_FALSE(read_dataset_file(dir / "missing.cvds"));
    test::write_text(dir / "text.cvds", "hello, this is not a dataset at all");
    EXPECT_FALSE(read_dataset_file(dir / "text.cvds"));
}

TEST(DatasetFile, RejectsATruncatedFile) {
    const test::TempDir dir;
    ASSERT_TRUE(write_dataset_file(dir / "pets.cvds", test::synthetic_set(4, 1)));
    std::filesystem::resize_file(dir / "pets.cvds", std::filesystem::file_size(dir / "pets.cvds") - 100);
    EXPECT_FALSE(read_dataset_file(dir / "pets.cvds"));
}

TEST(DatasetFile, RejectsALabelOfNoClass) {
    const test::TempDir dir;
    ASSERT_TRUE(write_dataset_file(dir / "pets.cvds", test::synthetic_set(2, 1)));
    {
        std::fstream file(dir / "pets.cvds", std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(20 + 2 * (4 + 7));  // the first label byte: after the 20 byte header and the two names "class 1", "class 2"
        file.put(7);
    }
    EXPECT_FALSE(read_dataset_file(dir / "pets.cvds"));
}

TEST(DatasetFile, RejectsPicturesOfAnotherSize) {
    const test::TempDir dir;
    ASSERT_TRUE(write_dataset_file(dir / "pets.cvds", {}));
    {
        std::fstream file(dir / "pets.cvds", std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(8);  // the side
        const std::uint32_t side = 32;
        file.write(reinterpret_cast<const char*>(&side), sizeof(side));
    }
    const auto loaded = read_dataset_file(dir / "pets.cvds");
    ASSERT_FALSE(loaded);
    EXPECT_NE(loaded.error().find("32x32"), std::string::npos);
}

}
