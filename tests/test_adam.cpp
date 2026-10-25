#include "model/adam.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace game {

TEST(Adam, FirstStepMovesEachParameterByAboutTheLearningRate) {
    // With the start-up correction the first step is lr * sign(gradient), whatever the gradient's size.
    Adam adam{3};
    std::vector<float> params{0.0f, 0.0f, 0.0f};
    adam.step(params, std::vector<float>{0.001f, -50.0f, 0.0f}, 0.1f);
    EXPECT_NEAR(params[0], -0.1f, 1e-4);
    EXPECT_NEAR(params[1], 0.1f, 1e-4);
    EXPECT_FLOAT_EQ(params[2], 0.0f);
}

TEST(Adam, FindsTheMinimumOfAParabola) {
    const std::vector<float> target{3.0f, -2.0f, 0.5f};
    std::vector<float> params{0.0f, 0.0f, 0.0f};
    Adam adam{3};
    for (int i = 0; i < 600; ++i) {
        std::vector<float> grads(3);
        for (std::size_t p = 0; p < 3; ++p) {
            grads[p] = 2.0f * (params[p] - target[p]);
        }
        adam.step(params, grads, 0.05f);
    }
    for (std::size_t p = 0; p < 3; ++p) {
        EXPECT_NEAR(params[p], target[p], 0.02f);
    }
}

}
