#include "model/model_file.h"
#include "model/model_store.h"

#include "test_support.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>

namespace game {

namespace {

constexpr CnnConfig kSmall{2, 3, 2, 5};

ModelFile small_model(std::uint32_t seed) {
    const Cnn cnn{kSmall, seed};
    ModelFile model;
    model.config = kSmall;
    model.epochs = 7;
    model.balanced_accuracy = 0.8125f;
    model.train_samples = 1234;
    model.split_known = true;
    model.split_seed = 987654321u;
    model.dataset_samples = 7390;
    model.class_names = {"cats", "dogs"};
    model.params.assign(cnn.params().begin(), cnn.params().end());
    return model;
}

}

TEST(ModelFile, RoundTripKeepsEverything) {
    const test::TempDir dir;
    const ModelFile model = small_model(1);
    ASSERT_TRUE(write_model_file(dir / "a.cvm", model));
    const auto loaded = read_model_file(dir / "a.cvm");
    ASSERT_TRUE(loaded) << loaded.error();
    EXPECT_EQ(loaded->config, model.config);
    EXPECT_EQ(loaded->epochs, 7u);
    EXPECT_FLOAT_EQ(loaded->balanced_accuracy, 0.8125f);
    EXPECT_EQ(loaded->train_samples, 1234u);
    EXPECT_TRUE(loaded->split_known);
    EXPECT_EQ(loaded->split_seed, 987654321u);
    EXPECT_EQ(loaded->dataset_samples, 7390u);
    EXPECT_EQ(loaded->params, model.params);
    EXPECT_EQ(loaded->class_names, model.class_names);
    EXPECT_FALSE(std::filesystem::exists(dir / "a.cvm.tmp"));
}

TEST(ModelFile, RoundTripKeepsTheNamesOfTheClasses) {
    const test::TempDir dir;
    ModelFile model = small_model(3);
    model.class_names = {"female", "male"};
    ASSERT_TRUE(write_model_file(dir / "people.cvm", model));
    const auto loaded = read_model_file(dir / "people.cvm");
    ASSERT_TRUE(loaded) << loaded.error();
    EXPECT_EQ(loaded->class_names, (ClassNames{"female", "male"}));
    EXPECT_EQ(loaded->params, model.params);
}

TEST(ModelFile, ReadsAVersion1FileWithoutASplit) {
    // Written by hand the way version 1 of the program wrote it: no split seed, no dataset size.
    const test::TempDir dir;
    const ModelFile model = small_model(6);
    std::vector<std::uint8_t> bytes{'C', 'V', 'M', 'D'};
    test::append(bytes, std::uint32_t{1});
    for (const int size : {model.config.conv1, model.config.conv2, model.config.conv3, model.config.hidden}) {
        test::append(bytes, static_cast<std::uint32_t>(size));
    }
    test::append(bytes, std::uint32_t{model.epochs});
    test::append(bytes, model.balanced_accuracy);
    test::append(bytes, std::uint32_t{model.train_samples});
    test::append(bytes, static_cast<std::uint32_t>(model.params.size()));
    for (const float value : model.params) {
        test::append(bytes, value);
    }
    test::write_bytes(dir / "old.cvm", bytes);

    const auto loaded = read_model_file(dir / "old.cvm");
    ASSERT_TRUE(loaded) << loaded.error();
    EXPECT_FALSE(loaded->split_known);
    EXPECT_EQ(loaded->class_names, (ClassNames{"cat", "dog"}));
    EXPECT_EQ(loaded->dataset_samples, 0u);
    EXPECT_EQ(loaded->epochs, model.epochs);
    EXPECT_EQ(loaded->params, model.params);
}

TEST(ModelFile, LoadedParametersBuildARunnableNetwork) {
    const test::TempDir dir;
    ASSERT_TRUE(write_model_file(dir / "a.cvm", small_model(2)));
    const auto loaded = read_model_file(dir / "a.cvm");
    ASSERT_TRUE(loaded);
    EXPECT_TRUE(Cnn::with_params(loaded->config, loaded->params));
}

TEST(ModelFile, RejectsMissingForeignAndTruncatedFiles) {
    const test::TempDir dir;
    EXPECT_FALSE(read_model_file(dir / "missing.cvm"));
    test::write_text(dir / "text.cvm", "this is not a model file at all, just words");
    EXPECT_FALSE(read_model_file(dir / "text.cvm"));

    ASSERT_TRUE(write_model_file(dir / "a.cvm", small_model(3)));
    std::filesystem::resize_file(dir / "a.cvm", std::filesystem::file_size(dir / "a.cvm") - 4);
    EXPECT_FALSE(read_model_file(dir / "a.cvm"));
}

TEST(ModelFile, RejectsAParameterCountThatDoesNotFitTheNetwork) {
    const test::TempDir dir;
    ModelFile model = small_model(4);
    model.params.push_back(1.0f);
    ASSERT_TRUE(write_model_file(dir / "a.cvm", model));
    EXPECT_FALSE(read_model_file(dir / "a.cvm"));
}

TEST(ModelFile, RejectsAnOutOfRangeNetworkSize) {
    const test::TempDir dir;
    ModelFile model = small_model(5);
    model.config.conv1 = 0;
    ASSERT_TRUE(write_model_file(dir / "a.cvm", model));
    EXPECT_FALSE(read_model_file(dir / "a.cvm"));
}

TEST(ModelStore, SavesListsLoadsAndRemoves) {
    const test::TempDir dir;
    const ModelStore store{dir / "models"};
    ASSERT_TRUE(store.save("b", small_model(1)));
    ASSERT_TRUE(store.save("a", small_model(2)));
    test::write_text(dir / "models" / "other.txt", "x");
    EXPECT_EQ(store.list(), (std::vector<std::string>{"a", "b"}));
    EXPECT_EQ(store.load("a").value().params, small_model(2).params);
    EXPECT_TRUE(store.remove("a"));
    EXPECT_FALSE(store.remove("a"));
    EXPECT_FALSE(store.load("a"));
    EXPECT_EQ(store.list(), (std::vector<std::string>{"b"}));
}

TEST(ModelStore, WithoutADirectoryThereIsNoStorage) {
    const ModelStore store{std::filesystem::path{}};
    EXPECT_FALSE(store.available());
    EXPECT_TRUE(store.list().empty());
    EXPECT_FALSE(store.save("x", small_model(1)));
    EXPECT_FALSE(store.remove("x"));
}

TEST(ModelStore, NewNamesCarryTheNetworkAndEpochsAndAreValidFileNames) {
    const std::string name = ModelStore::new_name(CnnConfig{8, 16, 32, 64}, 12);
    EXPECT_TRUE(name.starts_with("8-16-32-64_12ep_"));
    EXPECT_TRUE(std::ranges::all_of(name, [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_'; }));
}

}
