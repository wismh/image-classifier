#pragma once

#include <engine/ui/view_model.h>

namespace game {

// One row of a list of names: the saved models, the datasets.
class ListItem final : public engine::ui::ViewModel {
public:
    ListItem() {
        property(engine::ui::intern("label"), label);
        command(engine::ui::intern("select"), select);
    }

    engine::ui::Bindable<std::string> label;
    engine::ui::RelayCommand select;
};

}
