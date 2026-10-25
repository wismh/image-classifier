#pragma once

#include "view/list_item.h"

#include <engine/ui/view_model.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace game {

// A list of names on screen with one of them selected (marked with "> "): the saved networks, the datasets.
class NameList {
public:
    using List = engine::ui::BindableList<std::shared_ptr<ListItem>>;

    // `on_pick(index)` runs when the player clicks a row; without it a click only selects the row.
    explicit NameList(List& list, std::function<void(std::size_t)> on_pick = {})
        : list_(list)
        , on_pick_(std::move(on_pick)) {}

    NameList(const NameList&) = delete;
    NameList& operator=(const NameList&) = delete;

    // Shows `names` with nothing selected.
    void reset(std::vector<std::string> names);
    void select(std::size_t index);
    void clear_selection();

    [[nodiscard]] const std::vector<std::string>& names() const { return names_; }
    [[nodiscard]] bool contains(const std::string& name) const { return find(name).has_value(); }
    [[nodiscard]] std::optional<std::size_t> find(const std::string& name) const;
    [[nodiscard]] std::optional<std::size_t> selected() const { return selected_; }
    // The selected name; empty when there is none.
    [[nodiscard]] std::string selected_name() const { return selected_ ? names_[*selected_] : std::string{}; }

private:
    void update_labels();

    List& list_;
    std::function<void(std::size_t)> on_pick_;
    std::vector<std::string> names_;
    std::vector<std::shared_ptr<ListItem>> items_;
    std::optional<std::size_t> selected_;
};

}
