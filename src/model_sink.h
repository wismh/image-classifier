#pragma once

#include "model/model_file.h"

#include <string>

namespace game {

// Whoever puts a network to use: the training that just ended hands its network to it.
class ModelSink {
public:
    virtual ~ModelSink() = default;

    // Makes `model` (saved as `name`) the network in use.
    virtual void use_model(const std::string& name, const ModelFile& model) = 0;
};

}
