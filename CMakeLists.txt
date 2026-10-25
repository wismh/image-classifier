cmake_minimum_required(VERSION 3.20)
project(cats-vs-dogs LANGUAGES CXX)

# The Wind SDK. The editor passes CMAKE_PREFIX_PATH when it builds the game; CMakeUserPresets.json does for an IDE.
find_package(Wind REQUIRED)

add_subdirectory(src/model)

engine_add_game(cats-vs-dogs
    src/main.cpp
    src/game.h
    src/game.cpp
    src/status_line.h
    src/path_dialog.h
    src/path_dialog.cpp
    src/loc/tr.h
    src/loc/locale_prefs.h
    src/loc/locale_prefs.cpp
    src/normalizer.h
    src/normalizer.cpp
    src/dataset_manager.h
    src/dataset_manager.cpp
    src/train_settings.h
    src/train_settings.cpp
    src/trainer.h
    src/trainer.cpp
    src/model_library.h
    src/model_library.cpp
    src/recognizer.h
    src/recognizer.cpp
    src/ideal_dreamer.h
    src/ideal_dreamer.cpp
    src/filter_layers.h
    src/filter_dreamer.h
    src/filter_dreamer.cpp
    src/filter_panel.h
    src/filter_panel.cpp
    src/view/filter_paint.h
    src/view/filter_paint.cpp
    src/view/filter_row.h
    src/view/filter_row.cpp
    src/view/votes_paint.h
    src/view/votes_paint.cpp
    src/view/ideal_paint.h
    src/view/ideal_paint.cpp
    src/view/probability_paint.h
    src/view/probability_paint.cpp
    src/view/list_item.h
    src/view/name_list.h
    src/view/name_list.cpp
    src/view/format.h
    src/view/grid_row.h
    src/view/grid_row.cpp
    src/view/sample_paint.h
    src/view/sample_paint.cpp
    src/view/heat_paint.h
    src/view/heat_paint.cpp
    src/help/help.h
    src/help/help.cpp
    src/help/help_figures.h
    src/help/help_figures.cpp
    src/help/help_topic_table.h
    src/help/help_view_model.h
    src/view/pets_view_model.h
    src/view/pets_view_model.cpp)
target_include_directories(cats-vs-dogs PRIVATE src)
target_link_libraries(cats-vs-dogs PRIVATE pets_model)

# Unit tests of the domain. Off by default so that the editor build does not compile them.
option(CATS_VS_DOGS_TESTS "Build the unit tests" OFF)
if(CATS_VS_DOGS_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
