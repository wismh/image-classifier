# The domain: no engine types, so the game module and the tests link the same code. It is numeric code, so it is compiled
# with optimization in every configuration (the game's DebugGame builds with /Od, which makes training many times slower);
# the flags are set for this directory only, and debug information is kept.
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUGGAME "/MD /O2 /Ob2 /Zi")
else()
    set(CMAKE_CXX_FLAGS_DEBUGGAME "-O2 -g")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../kernels kernels)

add_library(pets_model STATIC
    pet_image.h
    binary_io.h
    parallel.h
    labeled_images.h
    image_decoder.h
    image_decoder.cpp
    image_resize.h
    image_resize.cpp
    source_scan.h
    source_scan.cpp
    normalization_run.h
    normalization_run.cpp
    dataset_file.h
    dataset_file.cpp
    dataset_store.h
    dataset_store.cpp
    dataset_split.h
    dataset_split.cpp
    layer.h
    conv_layer.h
    conv_layer.cpp
    relu_layer.h
    max_pool_layer.h
    max_pool_layer.cpp
    dense_layer.h
    dense_layer.cpp
    cnn_config.h
    cnn_config.cpp
    cnn.h
    cnn.cpp
    loss.h
    loss.cpp
    adam.h
    adam.cpp
    augment.h
    augment.cpp
    training_run.h
    training_run.cpp
    model_file.h
    model_file.cpp
    model_store.h
    model_store.cpp
    prediction.h
    prediction.cpp
    picture_file.h
    picture_file.cpp
    path_display.h
    path_display.cpp
    output_path.h
    output_path.cpp
    sample_order.h
    sample_order.cpp
    continuation.h
    continuation.cpp
    dream.h
    dream.cpp
    filter_view.h
    filter_view.cpp
    grad_cam.h
    grad_cam.cpp
    cell_image.h
    cell_image.cpp)
target_include_directories(pets_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(pets_model PUBLIC pets_kernels)
target_compile_features(pets_model PUBLIC cxx_std_23)
set_target_properties(pets_model PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    # The CRT of the game module (see engine_add_game), so the library links into it and into the tests.
    if(WIND_SDK_CONFIG STREQUAL "Debug")
        set_target_properties(pets_model PROPERTIES MSVC_RUNTIME_LIBRARY MultiThreadedDebugDLL)
    else()
        set_target_properties(pets_model PROPERTIES MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
    endif()
endif()
