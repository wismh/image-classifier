#include "model_library.h"

namespace game {

ModelLibrary::ModelLibrary(PetsViewModel& view, ModelStore store, const Tr& tr, StatusLine& status)
    : view_(view)
    , store_(std::move(store))
    , tr_(tr)
    , status_(status)
    , list_(view.models) {
    view_.deleteModel.bind_to<ModelLibrary, &ModelLibrary::delete_selected, &ModelLibrary::can_delete>(*this);
    rebuild({});
}

std::expected<ModelFile, std::string> ModelLibrary::load_selected() const {
    return store_.load(list_.selected_name());
}

std::expected<void, std::string> ModelLibrary::save(const std::string& name, const ModelFile& model) {
    if (auto saved = store_.save(name, model); !saved) {
        return saved;
    }
    rebuild(name);
    return {};
}

void ModelLibrary::delete_selected() {
    const std::string name = list_.selected_name();
    if (!store_.remove(name)) {
        status_.say("status.model_delete_failed", {{"name", name}});
        return;
    }
    rebuild({});
    status_.say("status.model_deleted", {{"name", name}});
}

// Lists the files again and selects `select_name`, or the first network when that is empty or not there.
void ModelLibrary::rebuild(const std::string& select_name) {
    list_.reset(store_.list());
    if (!list_.names().empty()) {
        list_.select(list_.find(select_name).value_or(0));
    }
}

}
