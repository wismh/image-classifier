#pragma once

#include "loc/tr.h"
#include "model/model_file.h"
#include "model/model_store.h"
#include "status_line.h"
#include "view/name_list.h"
#include "view/pets_view_model.h"

#include <expected>
#include <string>
#include <vector>

namespace game {

// The saved networks: the list on the Recognize tab with the selection in it, and the files behind it.
class ModelLibrary {
public:
    ModelLibrary(PetsViewModel& view, ModelStore store, const Tr& tr, StatusLine& status);

    ModelLibrary(const ModelLibrary&) = delete;
    ModelLibrary& operator=(const ModelLibrary&) = delete;

    [[nodiscard]] bool has_selection() const { return list_.selected().has_value(); }
    [[nodiscard]] bool contains(const std::string& name) const { return list_.contains(name); }
    // The name of the selected network; empty when there is none.
    [[nodiscard]] std::string selected_name() const { return list_.selected_name(); }

    // Reads the file of the selected network.
    [[nodiscard]] std::expected<ModelFile, std::string> load_selected() const;
    // Writes `model` under `name`, lists it and selects it.
    [[nodiscard]] std::expected<void, std::string> save(const std::string& name, const ModelFile& model);

private:
    void delete_selected();
    [[nodiscard]] bool can_delete() const { return has_selection(); }
    void rebuild(const std::string& select_name);

    PetsViewModel& view_;
    ModelStore store_;
    const Tr& tr_;
    StatusLine& status_;
    NameList list_;
};

}
