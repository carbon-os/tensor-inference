cmake_minimum_required(VERSION 3.21)

# Project name updated to tensor_inference
project(tensor_inference VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ─────────────────────────────────────────────────────────────
#  System / vcpkg-managed dependencies
#  Expects CMAKE_TOOLCHAIN_FILE to be passed or VCPKG_ROOT set
# ─────────────────────────────────────────────────────────────

find_package(OpenSSL        REQUIRED)
find_package(CURL           CONFIG REQUIRED)
find_package(nlohmann_json  CONFIG REQUIRED)
find_package(CUDAToolkit    REQUIRED)

# ─────────────────────────────────────────────────────────────
#  tensor::core
# ─────────────────────────────────────────────────────────────

add_library(tensor_core INTERFACE)
add_library(tensor::core ALIAS tensor_core)
target_include_directories(tensor_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ─────────────────────────────────────────────────────────────
#  tensor::parser
# ─────────────────────────────────────────────────────────────

add_library(tensor_parser STATIC
    src/parser/safetensors.cpp
    src/parser/config.cpp
    src/parser/weight_map.cpp
)
add_library(tensor::parser ALIAS tensor_parser)
target_include_directories(tensor_parser
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(tensor_parser
    PUBLIC  tensor::core
    PRIVATE nlohmann_json::nlohmann_json
)

# ─────────────────────────────────────────────────────────────
#  tensor::resolver
# ─────────────────────────────────────────────────────────────

add_library(tensor_resolver STATIC
    src/resolver/resolver.cpp
)
add_library(tensor::resolver ALIAS tensor_resolver)
target_include_directories(tensor_resolver
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(tensor_resolver
    PRIVATE
        CURL::libcurl
        nlohmann_json::nlohmann_json
)

# ─────────────────────────────────────────────────────────────
#  tensor::backend (CUDA)
# ─────────────────────────────────────────────────────────────

add_library(tensor_backend STATIC
    src/backend/cuda/device.cu
    src/backend/cuda/tensor.cu
    src/backend/cuda/ops.cu
)
add_library(tensor::backend ALIAS tensor_backend)
target_include_directories(tensor_backend
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(tensor_backend
    PUBLIC tensor::core
    PUBLIC CUDA::cublas
    PUBLIC CUDA::cudart
)
set_target_properties(tensor_backend PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES          "75;80;86;89;90"
)

# ─────────────────────────────────────────────────────────────
#  tensor::models
# ─────────────────────────────────────────────────────────────

add_library(tensor_models STATIC
    src/models/kv_cache.cpp
    src/models/llama/llama_config.cpp
    src/models/llama/llama_model.cpp
    src/models/qwen/qwen_model.cpp
    src/models/gemma/gemma_model.cpp
    src/models/mistral/mistral_model.cpp
)
add_library(tensor::models ALIAS tensor_models)
target_include_directories(tensor_models
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(tensor_models
    PUBLIC  tensor::core
    PUBLIC  tensor::parser
    PUBLIC  tensor::backend
    PRIVATE nlohmann_json::nlohmann_json
)

# ─────────────────────────────────────────────────────────────
#  tensor::tokenizer
# ─────────────────────────────────────────────────────────────

add_library(tensor_tokenizer STATIC
    src/tokenizer/tokenizer.cpp
)
add_library(tensor::tokenizer ALIAS tensor_tokenizer)
target_include_directories(tensor_tokenizer
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(tensor_tokenizer
    PUBLIC  tensor::core
    PRIVATE nlohmann_json::nlohmann_json
)

# ─────────────────────────────────────────────────────────────
#  tensor::inference
# ─────────────────────────────────────────────────────────────

add_library(tensor_inference_lib STATIC
    src/inference/generator.cpp
    src/inference/sampling/greedy.cpp
    src/inference/sampling/top_p.cpp
)
add_library(tensor::inference ALIAS tensor_inference_lib)
target_include_directories(tensor_inference_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(tensor_inference_lib
    PUBLIC tensor::core
    PUBLIC tensor::models
    PUBLIC tensor::tokenizer
    PUBLIC tensor::backend
)

# ─────────────────────────────────────────────────────────────
#  Tools
# ─────────────────────────────────────────────────────────────

option(TENSOR_BUILD_TOOLS "Build developer tools" OFF)

if(TENSOR_BUILD_TOOLS)
    # Helper macro for consistency
    macro(link_tensor_deps target_name)
        target_link_libraries(${target_name} PRIVATE
            tensor::parser
            tensor::backend
            tensor::models
            tensor::tokenizer
            tensor::inference
        )
    endmacro()

    add_executable(parser_safetensors tests/parser_safetensors.cpp)
    target_link_libraries(parser_safetensors PRIVATE tensor::parser)

    add_executable(model-cli tools/model-cli.cpp)
    target_link_libraries(model-cli PRIVATE tensor::resolver)

    add_executable(run_llama tools/run_llama.cpp)
    link_tensor_deps(run_llama)

    add_executable(run_qwen tools/run_qwen.cpp)
    link_tensor_deps(run_qwen)

    add_executable(run_gemma tools/run_gemma.cpp)
    link_tensor_deps(run_gemma)

    add_executable(run_mistral tools/run_mistral.cpp)
    link_tensor_deps(run_mistral)
endif()