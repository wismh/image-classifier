#pragma once

#include "loc/tr.h"
#include "model/dataset_store.h"
#include "view/pets_view_model.h"

#include <string>

namespace game {

// The line at the bottom of the window. It remembers which message it shows, not only the text, so a change of language
// can say the same thing again in the new language.
class StatusLine {
public:
    StatusLine(PetsViewModel& view, const Tr& tr)
        : view_(view)
        , tr_(tr) {}

    // A catalog message; `args` fill its `{name}` places.
    void say(std::string key, Tr::Args args = {}) {
        key_ = std::move(key);
        args_ = std::move(args);
        raw_.clear();
        refresh();
    }

    // Text with no translation, such as the reason a file could not be read.
    void say_raw(std::string text) {
        key_.clear();
        raw_ = std::move(text);
        refresh();
    }

    // Says what is wrong with a name the player typed. False (and nothing said) when the name is fine.
    bool reject_name(DatasetStore::NameProblem problem) {
        switch (problem) {
        case DatasetStore::NameProblem::None: return false;
        case DatasetStore::NameProblem::Empty: say("name.empty"); break;
        case DatasetStore::NameProblem::TooLong:
            say("name.too_long", {{"max", std::to_string(DatasetStore::kMaxNameLength)}});
            break;
        case DatasetStore::NameProblem::BadCharacters: say("name.bad_chars"); break;
        }
        return true;
    }

    void refresh() {
        view_.statusText = key_.empty() ? raw_ : tr_(key_, args_);
    }

private:
    PetsViewModel& view_;
    const Tr& tr_;
    std::string key_;
    Tr::Args args_;
    std::string raw_;
};

}
