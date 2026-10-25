#include "model/augment.h"
#include "model/dream.h"
#include "model/filter_view.h"

#include "test_support.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>

namespace game {

namespace {

constexpr CnnConfig kSmall{4, 9, 3, 8};

std::vector<std::uint8_t> noisy_picture(std::uint32_t seed) {
    std::mt19937 rng{seed};
    std::vector<std::uint8_t> picture(kImageBytes);
    for (auto& byte : picture) {
        byte = static_cast<std::uint8_t>(rng() % 256);
    }
    return picture;
}

// The mean output of one filter's convolution (before the ReLU) for a normalized picture.
float mean_convolution(const Cnn& model, int block, int filter, std::span<const std::uint8_t> picture) {
    std::vector<float> input(Cnn::kInputSize);
    to_input(picture, input);
    Cnn::Workspace workspace = model.make_workspace();
    const auto out = model.forward_until(input, workspace, Cnn::conv_layer(block));
    const std::size_t plane = static_cast<std::size_t>(Cnn::block_side(block)) * Cnn::block_side(block);
    return mean_activation(out.subspan(static_cast<std::size_t>(filter) * plane, plane));
}


// Every bias of the network set to zero: then each layer's answers reach the scores through functions that scale with
// them.
void zero_biases(Cnn& cnn) {
    std::size_t offset = 0;
    for (int block = 0; block < Cnn::kBlocks; ++block) {
        const std::size_t weights = static_cast<std::size_t>(cnn.filters(block)) * cnn.block_inputs(block) * 9;
        std::fill_n(cnn.params().begin() + static_cast<std::ptrdiff_t>(offset + weights), cnn.filters(block), 0.0f);
        offset += weights + static_cast<std::size_t>(cnn.filters(block));
    }
    const std::size_t hidden = static_cast<std::size_t>(cnn.hidden_neurons());
    std::fill_n(cnn.params().begin() + static_cast<std::ptrdiff_t>(offset + cnn.hidden_inputs() * hidden), hidden, 0.0f);
    offset += cnn.hidden_inputs() * hidden + hidden;
    std::fill_n(cnn.params().begin() + static_cast<std::ptrdiff_t>(offset + hidden * kNumClasses), kNumClasses, 0.0f);
}

std::size_t offset_of(const Cnn& cnn, std::span<const float> part) {
    return static_cast<std::size_t>(part.data() - cnn.params().data());
}
}

TEST(CnnBlocks, GeometryAndAccessorsFollowTheConfig) {
    const Cnn cnn{kSmall, 1};
    EXPECT_EQ(cnn.filters(0), 4);
    EXPECT_EQ(cnn.filters(1), 9);
    EXPECT_EQ(cnn.filters(2), 3);
    EXPECT_EQ(cnn.block_inputs(0), 3);
    EXPECT_EQ(cnn.block_inputs(1), 4);
    EXPECT_EQ(cnn.block_inputs(2), 9);
    EXPECT_EQ(Cnn::block_side(0), 64);
    EXPECT_EQ(Cnn::block_side(1), 32);
    EXPECT_EQ(Cnn::block_side(2), 16);
    EXPECT_EQ(cnn.conv_weights(0).size(), 4u * 3 * 9);
    EXPECT_EQ(cnn.conv_weights(1).size(), 9u * 4 * 9);
    EXPECT_EQ(cnn.conv_weights(2).size(), 3u * 9 * 9);
}

TEST(CnnBlocks, WeightsAndBiasesAreTheOnesTheNetworkRunsWith) {
    Cnn cnn{kSmall, 1};
    // The first convolution's parameters are the first in the vector: weights [filter][color][3][3], then the biases.
    cnn.params()[(2 * 3 + 1) * 9 + 4] = 7.0f;  // filter 2, green, the middle tap
    cnn.params()[4 * 3 * 9 + 3] = -2.5f;       // the bias of filter 3
    EXPECT_FLOAT_EQ(cnn.conv_weights(0)[(2 * 3 + 1) * 9 + 4], 7.0f);
    EXPECT_FLOAT_EQ(cnn.conv_bias(0, 3), -2.5f);
}

TEST(CnnBackwardUntil, GradientOfAMidLayerGoalMatchesNumericalOnes) {
    const Cnn cnn{kSmall, 3};
    std::mt19937 rng{5};
    std::uniform_real_distribution<float> uniform{-0.5f, 0.5f};
    std::vector<float> picture(Cnn::kInputSize);
    for (float& value : picture) {
        value = uniform(rng);
    }
    const std::size_t layer = Cnn::conv_layer(1);
    const std::size_t plane = 32 * 32;
    constexpr int kFilter = 5;
    // loss = the sum of one filter's output plane.
    std::vector<float> d_output(9 * plane, 0.0f);
    std::fill_n(d_output.begin() + kFilter * static_cast<std::ptrdiff_t>(plane), plane, 1.0f);

    Cnn::Workspace workspace = cnn.make_workspace(true);
    (void)cnn.forward_until(picture, workspace, layer);
    std::vector<float> grads(cnn.param_count(), 0.0f);
    cnn.backward_until(picture, workspace, layer, d_output, grads);
    const std::vector<float> analytic(workspace.input_gradient().begin(), workspace.input_gradient().end());

    const auto loss_at = [&](const std::vector<float>& input) {
        Cnn::Workspace plain = cnn.make_workspace();
        const auto out = cnn.forward_until(input, plain, layer);
        double sum = 0.0;
        for (std::size_t i = 0; i < plane; ++i) {
            sum += out[kFilter * plane + i];
        }
        return sum;
    };
    constexpr float kStep = 3e-4f;
    int checked = 0;
    int off = 0;
    for (std::size_t i = 0; i < picture.size(); i += 101) {
        std::vector<float> moved = picture;
        moved[i] += kStep;
        const double up = loss_at(moved);
        moved[i] -= 2.0f * kStep;
        const double numeric = (up - loss_at(moved)) / (2.0 * kStep);
        off += std::abs(analytic[i] - numeric) <= 2e-2 * std::max(0.05, std::abs(numeric)) ? 0 : 1;
        ++checked;
    }
    EXPECT_GT(checked, 100);
    EXPECT_LE(off, checked / 20) << off << " of " << checked << " pixels disagree";
}

TEST(FilterWeightsPicture, FirstBlockIsAColoredThreeByThree) {
    Cnn cnn{kSmall, 1};
    std::fill(cnn.params().begin(), cnn.params().begin() + 4 * 3 * 9, 0.0f);
    cnn.params()[(1 * 3 + 0) * 9 + 4] = 2.0f;   // filter 1, red, the middle: the largest weight
    cnn.params()[(1 * 3 + 2) * 9 + 0] = -1.0f;  // filter 1, blue, the top left corner
    const FilterPicture picture = filter_weights_picture(cnn, 0, 1);
    EXPECT_EQ(picture.kind, FilterPicture::Kind::Color);
    ASSERT_EQ(picture.width, 3);
    ASSERT_EQ(picture.height, 3);
    ASSERT_EQ(picture.values.size(), 27u);
    EXPECT_FLOAT_EQ(picture.values[4 * 3 + 0], 1.0f);   // the middle, red: the largest weight reaches the end of the range
    EXPECT_FLOAT_EQ(picture.values[4 * 3 + 1], 0.5f);   // green there is zero
    EXPECT_FLOAT_EQ(picture.values[0 * 3 + 2], 0.25f);  // the corner, blue: -1 / 2 of the way down from 0.5
}

TEST(FilterWeightsPicture, AFilterOfZerosIsNeutralGray) {
    Cnn cnn{kSmall, 1};
    std::fill(cnn.params().begin() + 3 * 9, cnn.params().begin() + 2 * 3 * 9, 0.0f);
    const FilterPicture picture = filter_weights_picture(cnn, 0, 1);
    for (const float value : picture.values) {
        EXPECT_FLOAT_EQ(value, 0.5f);
    }
}

TEST(FilterWeightsPicture, DeeperBlocksAreATileOfThreeByThreeForEveryInput) {
    const Cnn cnn{kSmall, 2};
    const FilterPicture picture = filter_weights_picture(cnn, 1, 0);  // 4 inputs: 2 x 2 tiles
    EXPECT_EQ(picture.kind, FilterPicture::Kind::Signed);
    EXPECT_EQ(picture.width, 2 * 3 + 1);
    EXPECT_EQ(picture.height, 2 * 3 + 1);
    ASSERT_EQ(picture.values.size(), 49u);
    // The gap between the tiles is zero, and the weights fill the range: the largest reaches 1 in size.
    for (int i = 0; i < 7; ++i) {
        EXPECT_EQ(picture.values[static_cast<std::size_t>(i * 7 + 3)], 0.0f);  // the column between the tiles
        EXPECT_EQ(picture.values[static_cast<std::size_t>(3 * 7 + i)], 0.0f);  // the row between them
    }
    float largest = 0.0f;
    for (const float value : picture.values) {
        EXPECT_GE(value, -1.0f);
        EXPECT_LE(value, 1.0f);
        largest = std::max(largest, std::abs(value));
    }
    EXPECT_FLOAT_EQ(largest, 1.0f);

    const FilterPicture nine = filter_weights_picture(cnn, 2, 0);  // 9 inputs: 3 x 3 tiles
    EXPECT_EQ(nine.width, 3 * 3 + 2);
    EXPECT_EQ(nine.height, 3 * 3 + 2);
    const FilterPicture three = filter_weights_picture(cnn, 1, 1);
    EXPECT_EQ(three.values.size(), 49u);
}

TEST(CnnDense, SizesAndBiasesFollowTheConfig) {
    Cnn cnn{kSmall, 1};
    EXPECT_EQ(cnn.hidden_neurons(), 8);
    EXPECT_EQ(cnn.hidden_inputs(), 3u * 8 * 8);
    EXPECT_EQ(cnn.hidden_weights().size(), 8u * 3 * 64);
    EXPECT_EQ(cnn.output_weights().size(), 2u * 8);
    cnn.params()[offset_of(cnn, cnn.hidden_weights()) + cnn.hidden_weights().size() + 5] = 3.5f;
    cnn.params()[offset_of(cnn, cnn.output_weights()) + cnn.output_weights().size() + 1] = -1.25f;
    EXPECT_FLOAT_EQ(cnn.hidden_bias(5), 3.5f);
    EXPECT_FLOAT_EQ(cnn.output_bias(1), -1.25f);
}

TEST(FilterVotes, WithoutBiasesEveryLayersVotesAddUpToTheGapBetweenTheScores) {
    // A network without biases scales with its input, so the answers times their gradients add up to the whole score.
    Cnn cnn{kSmall, 12};
    zero_biases(cnn);
    Cnn::Workspace workspace = cnn.make_workspace();
    for (std::uint32_t seed = 1; seed <= 3; ++seed) {
        const FilterVotes votes = run_filters(cnn, workspace, noisy_picture(seed));
        const float tolerance = 2e-3f * std::max(1.0f, std::abs(votes.score_gap));
        for (int block = 0; block < Cnn::kBlocks; ++block) {
            ASSERT_EQ(votes.conv[static_cast<std::size_t>(block)].size(), static_cast<std::size_t>(cnn.filters(block)));
            float sum = 0.0f;
            for (const float vote : votes.conv[static_cast<std::size_t>(block)]) {
                sum += vote;
            }
            EXPECT_NEAR(sum, votes.score_gap, tolerance) << "block " << block;
        }
        ASSERT_EQ(votes.hidden.size(), 8u);
        float hidden_sum = 0.0f;
        for (const float vote : votes.hidden) {
            hidden_sum += vote;
        }
        EXPECT_NEAR(hidden_sum, votes.score_gap, tolerance);
    }
}

TEST(FilterVotes, TheGapIsTheDogScoreMinusTheCatScore) {
    const Cnn cnn{kSmall, 3};
    Cnn::Workspace workspace = cnn.make_workspace();
    const FilterVotes votes = run_filters(cnn, workspace, noisy_picture(4));
    const auto& scores = workspace.outputs[Cnn::output_layer()];
    EXPECT_FLOAT_EQ(votes.score_gap, scores[1] - scores[0]);
    EXPECT_NEAR(class_probability(workspace, ClassId::First) + class_probability(workspace, ClassId::Second), 1.0f, 1e-6);
    EXPECT_EQ(class_probability(workspace, ClassId::Second) > 0.5f, votes.score_gap > 0.0f);
}

TEST(HiddenNeurons, WeightsPictureIsWhereTheNeuronLooksSummedOverTheMaps) {
    Cnn cnn{kSmall, 1};
    const std::size_t base = offset_of(cnn, cnn.hidden_weights());
    const std::size_t neuron_start = base + 1 * cnn.hidden_inputs();
    std::fill_n(cnn.params().begin() + static_cast<std::ptrdiff_t>(neuron_start), cnn.hidden_inputs(), 0.0f);
    cnn.params()[neuron_start + 0 * 64 + 5] = 2.0f;   // map 0, place 5
    cnn.params()[neuron_start + 2 * 64 + 5] = -1.0f;  // map 2, the same place: the two add up to 1
    cnn.params()[neuron_start + 1 * 64 + 9] = -4.0f;  // map 1, place 9: the largest sum in size
    const FilterPicture picture = hidden_weights_picture(cnn, 1);
    EXPECT_EQ(picture.kind, FilterPicture::Kind::Signed);
    ASSERT_EQ(picture.width, 8);
    ASSERT_EQ(picture.height, 8);
    ASSERT_EQ(picture.values.size(), 64u);
    EXPECT_FLOAT_EQ(picture.values[9], -1.0f);
    EXPECT_FLOAT_EQ(picture.values[5], 0.25f);
    EXPECT_FLOAT_EQ(picture.values[0], 0.0f);
}

TEST(HiddenNeurons, StrongestIsOneAndConnectionIsDogWeightMinusCatWeight) {
    Cnn cnn{kSmall, 5};
    float strongest = 0.0f;
    for (int neuron = 0; neuron < cnn.hidden_neurons(); ++neuron) {
        const float strength = hidden_strength(cnn, neuron);
        EXPECT_GE(strength, 0.0f);
        EXPECT_LE(strength, 1.0f);
        strongest = std::max(strongest, strength);
    }
    EXPECT_FLOAT_EQ(strongest, 1.0f);

    const std::size_t out = offset_of(cnn, cnn.output_weights());
    cnn.params()[out + 0 * 8 + 3] = 0.5f;  // the cat's weight from neuron 3
    cnn.params()[out + 1 * 8 + 3] = 2.0f;  // the dog's weight from neuron 3
    EXPECT_FLOAT_EQ(hidden_connection(cnn, 3), 1.5f);
}

TEST(HiddenNeurons, AnswerIsNeverNegative) {
    const Cnn cnn{kSmall, 6};
    Cnn::Workspace workspace = cnn.make_workspace();
    (void)run_filters(cnn, workspace, noisy_picture(2));
    for (int neuron = 0; neuron < cnn.hidden_neurons(); ++neuron) {
        EXPECT_GE(hidden_activation(workspace, neuron), 0.0f);
    }
}

TEST(OutputWeights, PictureIsTheSpeciesRowOfTheHiddenWeights) {
    Cnn cnn{kSmall, 7};
    const std::size_t out = offset_of(cnn, cnn.output_weights());
    for (std::size_t n = 0; n < 8; ++n) {
        cnn.params()[out + n] = 0.0f;      // the cat's row
        cnn.params()[out + 8 + n] = 0.0f;  // the dog's row
    }
    cnn.params()[out + 8 + 4] = -2.0f;  // the dog's weight from neuron 4
    cnn.params()[out + 8 + 1] = 1.0f;
    const FilterPicture dog = output_weights_picture(cnn, ClassId::Second);
    EXPECT_EQ(dog.width, 3);   // 8 neurons: 3 columns
    EXPECT_EQ(dog.height, 3);  // and 3 rows, the last place empty
    ASSERT_EQ(dog.values.size(), 9u);
    EXPECT_FLOAT_EQ(dog.values[4], -1.0f);
    EXPECT_FLOAT_EQ(dog.values[1], 0.5f);
    EXPECT_FLOAT_EQ(dog.values[8], 0.0f);
    for (const float value : output_weights_picture(cnn, ClassId::First).values) {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }
}

TEST(DreamHidden, MakesTheNeuronAnswerStrongerThanNoiseDoes) {
    const Cnn cnn{kSmall, 8};
    const std::atomic<bool> cancel{false};
    const int neuron = 2;
    const auto level = [&](std::span<const std::uint8_t> picture) {
        std::vector<float> input(Cnn::kInputSize);
        to_input(picture, input);
        Cnn::Workspace workspace = cnn.make_workspace();
        return cnn.forward_until(input, workspace, Cnn::hidden_layer())[static_cast<std::size_t>(neuron)];
    };
    const auto picture = dream_hidden(cnn, neuron, cancel);
    ASSERT_TRUE(picture);
    ASSERT_EQ(picture->size(), kImageBytes);
    float noise_level = -1e9f;
    for (std::uint32_t seed = 1; seed <= 5; ++seed) {
        noise_level = std::max(noise_level, level(noisy_picture(seed)));
    }
    EXPECT_GT(level(*picture), noise_level);
    EXPECT_EQ(dream_hidden(cnn, neuron, cancel), picture);
    EXPECT_NE(dream_hidden(cnn, 3, cancel), picture);
    const std::atomic<bool> stop{true};
    EXPECT_FALSE(dream_hidden(cnn, neuron, stop));
}

TEST(FilterStrength, StrongestFilterIsOneAndTheOthersAreLess) {
    const Cnn cnn{kSmall, 4};
    float strongest = 0.0f;
    for (int filter = 0; filter < cnn.filters(1); ++filter) {
        const float strength = filter_strength(cnn, 1, filter);
        EXPECT_GE(strength, 0.0f);
        EXPECT_LE(strength, 1.0f);
        strongest = std::max(strongest, strength);
    }
    EXPECT_FLOAT_EQ(strongest, 1.0f);
}

TEST(FilterMaps, AreTheReluOutputsOfEachBlockAndNeverNegative) {
    const Cnn cnn{kSmall, 6};
    Cnn::Workspace workspace = cnn.make_workspace();
    run_filters(cnn, workspace, noisy_picture(1));
    for (int block = 0; block < Cnn::kBlocks; ++block) {
        for (int filter = 0; filter < cnn.filters(block); ++filter) {
            const auto map = activation_map(workspace, block, filter);
            ASSERT_EQ(map.size(), static_cast<std::size_t>(Cnn::block_side(block)) * Cnn::block_side(block));
            EXPECT_TRUE(std::ranges::all_of(map, [](float v) { return v >= 0.0f; }));
        }
    }
    // Some filter of the first block fires on a noisy picture.
    float total = 0.0f;
    for (int filter = 0; filter < cnn.filters(0); ++filter) {
        total += mean_activation(activation_map(workspace, 0, filter));
    }
    EXPECT_GT(total, 0.0f);
}

TEST(FilterMaps, MeanActivationIsTheMeanOfTheMap) {
    EXPECT_FLOAT_EQ(mean_activation(std::vector<float>{1.0f, 2.0f, 3.0f, 6.0f}), 3.0f);
    EXPECT_FLOAT_EQ(mean_activation({}), 0.0f);
}

TEST(DreamFilter, MakesTheFilterAnswerFarStrongerThanNoiseDoes) {
    const Cnn cnn{kSmall, 8};
    const std::atomic<bool> cancel{false};
    for (const int block : {0, 1, 2}) {
        const int filter = block;  // any
        const auto picture = dream_filter(cnn, block, filter, cancel);
        ASSERT_TRUE(picture);
        ASSERT_EQ(picture->size(), kImageBytes);
        float noise_level = 0.0f;
        for (std::uint32_t seed = 1; seed <= 5; ++seed) {
            noise_level = std::max(noise_level, mean_convolution(cnn, block, filter, noisy_picture(seed)));
        }
        // A fresh network's filters are small, so the margin is too: what counts is that the dream clears every noise
        // picture, which averages out to about nothing.
        EXPECT_GT(mean_convolution(cnn, block, filter, *picture), noise_level + 0.005f) << "block " << block;
    }
}

TEST(DreamFilter, TwoFiltersGetDifferentPicturesAndTheSameFilterTheSameOne) {
    const Cnn cnn{kSmall, 8};
    const std::atomic<bool> cancel{false};
    EXPECT_NE(dream_filter(cnn, 0, 0, cancel), dream_filter(cnn, 0, 1, cancel));
    EXPECT_EQ(dream_filter(cnn, 1, 2, cancel), dream_filter(cnn, 1, 2, cancel));
}

TEST(DreamFilter, StopsWhenCancelled) {
    const Cnn cnn{kSmall, 8};
    const std::atomic<bool> cancel{true};
    EXPECT_FALSE(dream_filter(cnn, 0, 0, cancel));
}

}
