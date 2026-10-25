# Unit tests of the domain (src/model). They link the same pets_model library as the game module and GTest from the
# SDK; no engine code is compiled or linked.
add_executable(pets_tests
    test_support.h
    test_adam.cpp
    test_augment.cpp
    test_cell_image.cpp
    test_cnn.cpp
    test_dataset_file.cpp
    test_dream.cpp
    test_filter_view.cpp
    test_grad_cam.cpp
    test_gemm.cpp
    test_dataset_split.cpp
    test_dataset_store.cpp
    test_image_resize.cpp
    test_layers.cpp
    test_loss.cpp
    test_model_file.cpp
    test_normalization_run.cpp
    test_path_display.cpp
    test_output_path.cpp
    test_sample_order.cpp
    test_continuation.cpp
    test_prediction.cpp
    test_source_scan.cpp
    test_training_run.cpp)
target_link_libraries(pets_tests PRIVATE pets_model GTest::gtest_main)
if(MSVC)
    if(WIND_SDK_CONFIG STREQUAL "Debug")
        set_target_properties(pets_tests PROPERTIES MSVC_RUNTIME_LIBRARY MultiThreadedDebugDLL)
    else()
        set_target_properties(pets_tests PROPERTIES MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
    endif()
endif()
gtest_discover_tests(pets_tests)
