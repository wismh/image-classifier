# The matrix kernels of the network. They are compiled with optimization in every configuration: the game's DebugGame
# builds with /Od, and a convolution through unoptimized intrinsics is far slower than plain loops. The flags are set for
# this directory only (src/model sets the same for the rest of the domain).
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUGGAME "/MD /O2 /Ob2 /Zi")
else()
    set(CMAKE_CXX_FLAGS_DEBUGGAME "-O2 -g")
endif()

add_library(pets_kernels STATIC
    gemm.h
    gemm_internal.h
    layer_ops.h
    layer_ops.cpp
    vector_ops.h
    vector_ops.cpp
    gemm.cpp
    gemm_generic.cpp)
target_include_directories(pets_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(pets_kernels PUBLIC cxx_std_23)
set_target_properties(pets_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The AVX2/FMA kernels exist only in an x86-64 build; elsewhere gemm.cpp falls back to the plain loops.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64)$")
    target_sources(pets_kernels PRIVATE gemm_avx2.cpp)
    target_compile_definitions(pets_kernels PRIVATE PETS_HAVE_AVX2_KERNELS)
    if(MSVC)
        set_source_files_properties(gemm_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(gemm_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
endif()

if(MSVC)
    # The CRT of the game module (see engine_add_game), so the library links into it and into the tests.
    if(WIND_SDK_CONFIG STREQUAL "Debug")
        set_target_properties(pets_kernels PROPERTIES MSVC_RUNTIME_LIBRARY MultiThreadedDebugDLL)
    else()
        set_target_properties(pets_kernels PROPERTIES MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
    endif()
endif()
