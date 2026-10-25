#include "model/dataset_store.h"

#include "model/dataset_file.h"

#include "test_support.h"

#include <gtest/gtest.h>

namespace game {

TEST(DatasetStoreName, AcceptsPlainNames) {
    EXPECT_EQ(DatasetStore::check_name("pets"), DatasetStore::NameProblem::None);
    EXPECT_EQ(DatasetStore::check_name("Oxford_2-b"), DatasetStore::NameProblem::None);
    EXPECT_EQ(DatasetStore::check_name(std::string(DatasetStore::kMaxNameLength, 'a')), DatasetStore::NameProblem::None);
}

TEST(DatasetStoreName, RejectsBadNames) {
    EXPECT_EQ(DatasetStore::check_name(""), DatasetStore::NameProblem::Empty);
    EXPECT_EQ(DatasetStore::check_name(std::string(DatasetStore::kMaxNameLength + 1, 'a')),
            DatasetStore::NameProblem::TooLong);
    EXPECT_EQ(DatasetStore::check_name("a b"), DatasetStore::NameProblem::BadCharacters);
    EXPECT_EQ(DatasetStore::check_name("../x"), DatasetStore::NameProblem::BadCharacters);
    EXPECT_EQ(DatasetStore::check_name("a.b"), DatasetStore::NameProblem::BadCharacters);
}

TEST(DatasetStoreName, SuggestionIsAlwaysAValidName) {
    EXPECT_EQ(DatasetStore::suggest_name("C:/data/oxford pets.v2.cvds"), "oxford_pets_v2");
    EXPECT_EQ(DatasetStore::suggest_name("pets.cvds"), "pets");
    const std::string suggested = DatasetStore::suggest_name(std::string(80, 'x') + ".cvds");
    EXPECT_EQ(suggested.size(), DatasetStore::kMaxNameLength);
    EXPECT_EQ(DatasetStore::check_name(suggested), DatasetStore::NameProblem::None);
    EXPECT_EQ(DatasetStore::check_name(DatasetStore::suggest_name("")), DatasetStore::NameProblem::Empty);
}

TEST(DatasetStore, RoundTripKeepsLabelsAndPixels) {
    const test::TempDir dir;
    const DatasetStore store{dir / "sets"};
    const LabeledImages data = test::synthetic_set(4, 1);
    ASSERT_TRUE(store.save("mine", data));
    EXPECT_TRUE(store.contains("mine"));

    const auto loaded = store.load("mine");
    ASSERT_TRUE(loaded) << loaded.error();
    EXPECT_EQ(loaded->labels, data.labels);
    EXPECT_EQ(loaded->images, data.images);
}

TEST(DatasetStore, ListsNamesSortedAndRemoves) {
    const test::TempDir dir;
    const DatasetStore store{dir.path()};
    ASSERT_TRUE(store.save("b", {}));
    ASSERT_TRUE(store.save("a", {}));
    test::write_text(dir / "other.txt", "x");
    EXPECT_EQ(store.list(), (std::vector<std::string>{"a", "b"}));
    EXPECT_TRUE(store.remove("a"));
    EXPECT_FALSE(store.remove("a"));
    EXPECT_FALSE(store.contains("a"));
    EXPECT_EQ(store.list(), (std::vector<std::string>{"b"}));
}

TEST(DatasetStore, AddsANormalizedFileFromAnywhere) {
    const test::TempDir dir;
    const LabeledImages data = test::synthetic_set(6, 2);
    ASSERT_TRUE(write_dataset_file(dir / "downloads" / "shared.cvds", data));

    const DatasetStore store{dir / "sets"};
    const auto added = store.add_file(dir / "downloads" / "shared.cvds", "shared");
    ASSERT_TRUE(added) << added.error();
    EXPECT_EQ(*added, 6u);
    EXPECT_EQ(store.load("shared").value().images, data.images);
    EXPECT_TRUE(std::filesystem::exists(dir / "downloads" / "shared.cvds"));
}

TEST(DatasetStore, AddingABadFileStoresNothing) {
    const test::TempDir dir;
    test::write_text(dir / "bad.cvds", "garbage");
    const DatasetStore store{dir / "sets"};
    EXPECT_FALSE(store.add_file(dir / "bad.cvds", "bad"));
    EXPECT_FALSE(store.add_file(dir / "missing.cvds", "missing"));
    EXPECT_TRUE(store.list().empty());
}

TEST(DatasetStore, WithoutADirectoryThereIsNoStorage) {
    const DatasetStore store{std::filesystem::path{}};
    EXPECT_FALSE(store.available());
    EXPECT_TRUE(store.list().empty());
    EXPECT_FALSE(store.contains("x"));
    EXPECT_FALSE(store.save("x", {}));
    EXPECT_FALSE(store.remove("x"));
}

}
