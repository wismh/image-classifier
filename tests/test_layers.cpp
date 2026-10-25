#include "model/conv_layer.h"
#include "model/dense_layer.h"
#include "model/max_pool_layer.h"
#include "model/relu_layer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <thread>
#include <vector>

namespace game {

namespace {

std::vector<float> random_values(std::size_t count, std::uint32_t seed, float spread = 1.0f) {
    std::mt19937 rng{seed};
    std::normal_distribution<float> normal{0.0f, spread};
    std::vector<float> values(count);
    for (float& value : values) {
        value = normal(rng);
    }
    return values;
}

std::vector<float> run_forward(const Layer& layer, const std::vector<float>& params, const std::vector<float>& input) {
    std::vector<float> out(layer.output_size());
    layer.forward(params, input, out);
    return out;
}

// loss = sum(output * weights): linear in the output, so its gradient with respect to the output is `weights`.
double loss_of(const Layer& layer, const std::vector<float>& params, const std::vector<float>& input,
        const std::vector<float>& weights) {
    const std::vector<float> out = run_forward(layer, params, input);
    double sum = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        sum += static_cast<double>(out[i]) * weights[i];
    }
    return sum;
}

// Compares the gradients the layer reports with central differences of the loss, for every input value and every
// parameter.
void check_gradients(const Layer& layer, std::uint32_t seed, float keep_away_from_zero = 0.0f) {
    std::vector<float> params = random_values(layer.param_count(), seed, 0.7f);  // biases too, not only weights
    std::vector<float> input = random_values(layer.input_size(), seed + 1);
    for (float& value : input) {  // a step that crosses a kink (ReLU at 0) has no derivative to compare with
        if (std::abs(value) < keep_away_from_zero) {
            value = value < 0.0f ? -keep_away_from_zero : keep_away_from_zero;
        }
    }
    const std::vector<float> weights = random_values(layer.output_size(), seed + 2);

    const std::vector<float> out = run_forward(layer, params, input);
    std::vector<float> d_in(layer.input_size(), 123.0f);  // stale values the layer must overwrite
    std::vector<float> grad_params(layer.param_count(), 0.0f);
    layer.backward(params, input, out, weights, d_in, grad_params);

    constexpr float kStep = 1e-2f;
    const auto check = [&](std::vector<float>& values, float analytic, std::size_t index, const char* what) {
        const float original = values[index];
        values[index] = original + kStep;
        const double up = loss_of(layer, params, input, weights);
        values[index] = original - kStep;
        const double down = loss_of(layer, params, input, weights);
        values[index] = original;
        const double numeric = (up - down) / (2.0 * kStep);
        EXPECT_NEAR(analytic, numeric, 2e-3 * std::max(1.0, std::abs(numeric))) << what << " " << index;
    };
    for (std::size_t i = 0; i < input.size(); ++i) {
        check(input, d_in[i], i, "input");
    }
    for (std::size_t i = 0; i < params.size(); ++i) {
        check(params, grad_params[i], i, "param");
    }
}

// The convolution as the textbook loops, the reference the matrix-product version is checked against.
struct ConvShape {
    int in_channels;
    int out_channels;
    int height;
    int width;
};

std::vector<float> reference_conv_forward(const ConvShape& s, const std::vector<float>& params,
        const std::vector<float>& input) {
    std::vector<float> out(static_cast<std::size_t>(s.out_channels) * s.height * s.width);
    const std::size_t weights = static_cast<std::size_t>(s.out_channels) * s.in_channels * 9;
    for (int oc = 0; oc < s.out_channels; ++oc) {
        for (int y = 0; y < s.height; ++y) {
            for (int x = 0; x < s.width; ++x) {
                double sum = params[weights + static_cast<std::size_t>(oc)];
                for (int ic = 0; ic < s.in_channels; ++ic) {
                    for (int ky = 0; ky < 3; ++ky) {
                        for (int kx = 0; kx < 3; ++kx) {
                            const int sy = y + ky - 1;
                            const int sx = x + kx - 1;
                            if (sy < 0 || sy >= s.height || sx < 0 || sx >= s.width) {
                                continue;
                            }
                            sum += static_cast<double>(params[((static_cast<std::size_t>(oc) * s.in_channels + ic) * 3 + ky) * 3 + kx])
                                    * input[(static_cast<std::size_t>(ic) * s.height + sy) * s.width + sx];
                        }
                    }
                }
                out[(static_cast<std::size_t>(oc) * s.height + y) * s.width + x] = static_cast<float>(sum);
            }
        }
    }
    return out;
}

// d_in and the parameter gradients for loss = sum(out * d_out).
void reference_conv_backward(const ConvShape& s, const std::vector<float>& params, const std::vector<float>& input,
        const std::vector<float>& d_out, std::vector<double>& d_in, std::vector<double>& grads) {
    const std::size_t weights = static_cast<std::size_t>(s.out_channels) * s.in_channels * 9;
    d_in.assign(input.size(), 0.0);
    grads.assign(params.size(), 0.0);
    for (int oc = 0; oc < s.out_channels; ++oc) {
        for (int y = 0; y < s.height; ++y) {
            for (int x = 0; x < s.width; ++x) {
                const double delta = d_out[(static_cast<std::size_t>(oc) * s.height + y) * s.width + x];
                grads[weights + static_cast<std::size_t>(oc)] += delta;
                for (int ic = 0; ic < s.in_channels; ++ic) {
                    for (int ky = 0; ky < 3; ++ky) {
                        for (int kx = 0; kx < 3; ++kx) {
                            const int sy = y + ky - 1;
                            const int sx = x + kx - 1;
                            if (sy < 0 || sy >= s.height || sx < 0 || sx >= s.width) {
                                continue;
                            }
                            const std::size_t w = ((static_cast<std::size_t>(oc) * s.in_channels + ic) * 3 + ky) * 3 + kx;
                            const std::size_t i = (static_cast<std::size_t>(ic) * s.height + sy) * s.width + sx;
                            grads[w] += delta * input[i];
                            d_in[i] += delta * params[w];
                        }
                    }
                }
            }
        }
    }
}

}

TEST(ConvLayer, CenterTapOnlyCopiesTheInputAndAddsTheBias) {
    const ConvLayer layer{1, 1, 3, 4};
    std::vector<float> params(layer.param_count(), 0.0f);
    params[4] = 1.0f;   // the middle of the 3x3 kernel
    params[9] = 0.5f;   // the bias
    const std::vector<float> input = random_values(12, 1);
    const std::vector<float> out = run_forward(layer, params, input);
    for (std::size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(out[i], input[i] + 0.5f);
    }
}

TEST(ConvLayer, CornerTapShiftsTheInputAndPadsWithZeros) {
    const ConvLayer layer{1, 1, 3, 3};
    std::vector<float> params(layer.param_count(), 0.0f);
    params[0] = 1.0f;  // kernel[0][0]: the output at (y, x) sees the input at (y - 1, x - 1)
    const std::vector<float> input{1, 2, 3, 4, 5, 6, 7, 8, 9};
    const std::vector<float> out = run_forward(layer, params, input);
    EXPECT_EQ(out, (std::vector<float>{0, 0, 0, 0, 1, 2, 0, 4, 5}));
}

TEST(ConvLayer, SumsOverInputChannelsAndKeepsOutputChannelsApart) {
    const ConvLayer layer{2, 2, 2, 2};
    std::vector<float> params(layer.param_count(), 0.0f);
    const auto tap = [&](int oc, int ic) -> float& { return params[static_cast<std::size_t>((oc * 2 + ic) * 9 + 4)]; };
    tap(0, 0) = 1.0f;
    tap(0, 1) = 10.0f;
    tap(1, 1) = 100.0f;
    const std::vector<float> input{1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<float> out = run_forward(layer, params, input);
    EXPECT_EQ(out, (std::vector<float>{51, 62, 73, 84, 500, 600, 700, 800}));
}

TEST(ConvLayer, GradientsMatchNumericalOnes) {
    check_gradients(ConvLayer{2, 3, 5, 6}, 10);
}

TEST(ConvLayer, MatchesTheTextbookLoopsOnTheSizesTheNetworkUses) {
    const ConvShape shapes[] = {{3, 8, 64, 64}, {8, 16, 32, 32}, {16, 32, 16, 16}, {2, 3, 5, 6}, {1, 1, 3, 3},
            {4, 5, 7, 9}, {3, 2, 8, 24}};
    for (const ConvShape& shape : shapes) {
        const ConvLayer layer{shape.in_channels, shape.out_channels, shape.height, shape.width};
        const std::vector<float> params = random_values(layer.param_count(), 100 + shape.height, 0.5f);
        const std::vector<float> input = random_values(layer.input_size(), 200 + shape.width);
        const std::vector<float> d_out = random_values(layer.output_size(), 300 + shape.out_channels);

        const std::vector<float> out = run_forward(layer, params, input);
        const std::vector<float> expected = reference_conv_forward(shape, params, input);
        for (std::size_t i = 0; i < out.size(); ++i) {
            ASSERT_NEAR(out[i], expected[i], 2e-4 * std::max(1.0f, std::abs(expected[i])))
                    << shape.height << "x" << shape.width << " at " << i;
        }

        std::vector<float> d_in(layer.input_size(), 7.0f);
        std::vector<float> grads(layer.param_count(), 0.0f);
        layer.backward(params, input, out, d_out, d_in, grads);
        std::vector<double> expected_d_in;
        std::vector<double> expected_grads;
        reference_conv_backward(shape, params, input, d_out, expected_d_in, expected_grads);
        for (std::size_t i = 0; i < d_in.size(); ++i) {
            ASSERT_NEAR(d_in[i], expected_d_in[i], 5e-4 * std::max(1.0, std::abs(expected_d_in[i])))
                    << shape.height << "x" << shape.width << " d_in " << i;
        }
        for (std::size_t i = 0; i < grads.size(); ++i) {
            ASSERT_NEAR(grads[i], expected_grads[i], 5e-4 * std::max(1.0, std::abs(expected_grads[i])))
                    << shape.height << "x" << shape.width << " grad " << i;
        }
    }
}

TEST(ConvLayer, ThreadsDoNotDisturbEachOther) {
    // The working space is per thread: layers shared by many threads at once give what one thread gives.
    const ConvLayer layer{3, 8, 16, 16};
    const std::vector<float> params = random_values(layer.param_count(), 1, 0.5f);
    const std::vector<float> input = random_values(layer.input_size(), 2);
    const std::vector<float> expected = run_forward(layer, params, input);

    std::vector<std::vector<float>> results(4);
    {
        std::vector<std::jthread> threads;
        for (std::size_t t = 0; t < results.size(); ++t) {
            threads.emplace_back([&, t] {
                for (int repeat = 0; repeat < 50; ++repeat) {
                    results[t] = run_forward(layer, params, input);
                }
            });
        }
    }
    for (const auto& result : results) {
        EXPECT_EQ(result, expected);
    }
}

TEST(ConvLayer, SkipsTheInputGradientWhenAskedTo) {
    const ConvLayer layer{2, 3, 4, 4};
    const std::vector<float> params = random_values(layer.param_count(), 1);
    const std::vector<float> input = random_values(layer.input_size(), 2);
    const std::vector<float> weights = random_values(layer.output_size(), 3);
    const std::vector<float> out = run_forward(layer, params, input);

    std::vector<float> with_input(layer.param_count(), 0.0f);
    std::vector<float> without_input(layer.param_count(), 0.0f);
    std::vector<float> d_in(layer.input_size());
    layer.backward(params, input, out, weights, d_in, with_input);
    layer.backward(params, input, out, weights, {}, without_input);
    EXPECT_EQ(with_input, without_input);
}

TEST(ConvLayer, ParameterGradientsAreAddedNotReplaced) {
    const ConvLayer layer{1, 2, 3, 3};
    const std::vector<float> params = random_values(layer.param_count(), 1);
    const std::vector<float> input = random_values(layer.input_size(), 2);
    const std::vector<float> weights = random_values(layer.output_size(), 3);
    const std::vector<float> out = run_forward(layer, params, input);

    std::vector<float> once(layer.param_count(), 0.0f);
    layer.backward(params, input, out, weights, {}, once);
    std::vector<float> twice = once;
    layer.backward(params, input, out, weights, {}, twice);
    for (std::size_t i = 0; i < once.size(); ++i) {
        EXPECT_FLOAT_EQ(twice[i], 2.0f * once[i]);
    }
}

TEST(ConvLayer, SizesAndParameterCount) {
    const ConvLayer layer{3, 8, 64, 64};
    EXPECT_EQ(layer.input_size(), 3u * 64 * 64);
    EXPECT_EQ(layer.output_size(), 8u * 64 * 64);
    EXPECT_EQ(layer.param_count(), 8u * 3 * 9 + 8);
}

TEST(ConvLayer, InitGivesSpreadWeightsAndZeroBiases) {
    const ConvLayer layer{3, 8, 8, 8};
    std::vector<float> params(layer.param_count(), 7.0f);
    std::mt19937 rng{1};
    layer.init(params, rng);
    const std::size_t weights = 8 * 3 * 9;
    double sum_squares = 0.0;
    for (std::size_t i = 0; i < weights; ++i) {
        sum_squares += static_cast<double>(params[i]) * params[i];
    }
    const double deviation = std::sqrt(sum_squares / weights);
    EXPECT_NEAR(deviation, std::sqrt(2.0 / 27.0), 0.04);
    for (std::size_t i = weights; i < params.size(); ++i) {
        EXPECT_EQ(params[i], 0.0f);
    }
}

TEST(ReluLayer, ClampsNegativesToZero) {
    const ReluLayer layer{5};
    EXPECT_EQ(run_forward(layer, {}, {-2, -0.5f, 0, 0.5f, 2}), (std::vector<float>{0, 0, 0, 0.5f, 2}));
}

TEST(ReluLayer, GradientsMatchNumericalOnes) {
    check_gradients(ReluLayer{40}, 20, 0.05f);
}

TEST(MaxPoolLayer, TakesTheLargestOfEachBlockPerChannel) {
    const MaxPoolLayer layer{2, 2, 4};
    const std::vector<float> input{
            1, 5, 2, 0,
            3, 4, 9, 8,
            -1, -2, -3, -4,
            -5, -6, -7, -8};
    EXPECT_EQ(run_forward(layer, {}, input), (std::vector<float>{5, 9, -1, -3}));
}

TEST(MaxPoolLayer, GradientGoesToTheLargestValueOnly) {
    const MaxPoolLayer layer{1, 2, 4};
    const std::vector<float> input{1, 5, 2, 0, 3, 4, 9, 8};
    const std::vector<float> out = run_forward(layer, {}, input);
    std::vector<float> d_in(8, 99.0f);
    layer.backward({}, input, out, std::vector<float>{10, 20}, d_in, {});
    EXPECT_EQ(d_in, (std::vector<float>{0, 10, 0, 0, 0, 0, 20, 0}));
}

TEST(MaxPoolLayer, TieGoesToTheFirstValueOfTheBlock) {
    const MaxPoolLayer layer{1, 2, 2};
    const std::vector<float> input{4, 4, 4, 4};
    const std::vector<float> out = run_forward(layer, {}, input);
    std::vector<float> d_in(4);
    layer.backward({}, input, out, std::vector<float>{1}, d_in, {});
    EXPECT_EQ(d_in, (std::vector<float>{1, 0, 0, 0}));
}

TEST(MaxPoolLayer, GradientsMatchNumericalOnes) {
    check_gradients(MaxPoolLayer{3, 4, 6}, 30);
}

TEST(DenseLayer, ComputesWeightedSumsPlusBias) {
    const DenseLayer layer{3, 2};
    // weights: row 0 = (1, 2, 3), row 1 = (-1, 0, 1); biases: 10, 20
    const std::vector<float> params{1, 2, 3, -1, 0, 1, 10, 20};
    EXPECT_EQ(run_forward(layer, params, {1, 1, 2}), (std::vector<float>{19, 21}));
}

TEST(DenseLayer, GradientsMatchNumericalOnes) {
    check_gradients(DenseLayer{7, 4}, 40);
}

TEST(DenseLayer, ParameterCountAndInit) {
    const DenseLayer layer{2048, 64};
    EXPECT_EQ(layer.param_count(), 2048u * 64 + 64);
    std::vector<float> params(layer.param_count(), 5.0f);
    std::mt19937 rng{2};
    layer.init(params, rng);
    EXPECT_EQ(params.back(), 0.0f);
    EXPECT_NE(params.front(), params[1]);
}

}
