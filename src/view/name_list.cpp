#include "view/name_list.h"

namespace game {

void NameList::reset(std::vector<std::string> names) {
    names_ = std::move(names);
    selected_.reset();
    items_.clear();
    for (std::size_t i = 0; i < names_.size(); ++i) {
        auto item = std::make_shared<ListItem>();
        item->select = [this, i] {
            if (on_pick_) {
                on_pick_(i);
            } else {
                select(i);
            }
        };
        items_.push_back(std::move(item));
    }
    update_labels();
    list_.set(items_);
}

void NameList::select(std::size_t index) {
    selected_ = index;
    update_labels();
}

void NameList::clear_selection() {
    selected_.reset();
    update_labels();
}

std::optional<std::size_t> NameList::find(const std::string& name) const {
    const auto found = std::ranges::find(names_, name);
    if (found == names_.end()) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(found - names_.begin());
}

void NameList::update_labels() {
    for (std::size_t i = 0; i < items_.size(); ++i) {
        items_[i]->label = (selected_ == i ? "> " : "   ") + names_[i];
    }
}

}
