#pragma once

#include "model/cnn.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace game {

// Where in a picture the network found evidence for each species (Grad-CAM). The last block's maps are weighed by how
// much the species' lead over the other species grows with them (its score minus the other's), and added up: a place
// that the maps which favor the species are active at comes out high. A place argues for one species or the other,
// never
// both. The maps are block_side(kBlocks - 1) x block_side(kBlocks - 1) places of the picture, seen from far.
struct GradCam {
    int side = 0;
    // [species]: side * side values in 0..1, both scaled by the same number so that the strongest place of the two is 1
    // and the species the picture does not speak for stays faint. All zero when no place argues for either.
    std::array<std::vector<float>, kNumClasses> maps;
};

// Runs the network on a normalized picture (kImageBytes bytes) and works out the maps. Leaves the run in `workspace`.
[[nodiscard]] GradCam grad_cam(const Cnn& model, Cnn::Workspace& workspace, std::span<const std::uint8_t> picture);

}
