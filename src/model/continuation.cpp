#include "model/continuation.h"

namespace game {

Continuation plan_continuation(const ModelFile& file, const LabeledImages& data, std::uint32_t fresh_seed) {
    Continuation plan;
    plan.split_seed = file.split_known ? file.split_seed : fresh_seed;
    if (file.class_names != data.class_names) {
        plan.kind = ContinueKind::OtherClasses;
    } else if (!file.split_known) {
        plan.kind = ContinueKind::UnknownSplit;
    } else if (file.dataset_samples != data.size()) {
        plan.kind = ContinueKind::OtherDataset;
    }
    return plan;
}

}
