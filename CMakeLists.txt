cmake_minimum_required(VERSION 3.21)
project(libtransformers VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ─────────────────────────────────────────────────────────────
#  System / vcpkg-managed dependencies
#  Pass -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake
#  (or set VCPKG_ROOT and let the preset handle it).
# ─────────────────────────────────────────────────────────────

find_package(OpenSSL        REQUIRED)
find_package(CURL           CONFIG REQUIRED)
find_package(nlohmann_json  CONFIG REQUIRED)
find_package(CUDAToolkit    REQUIRED)

# ─────────────────────────────────────────────────────────────
#  transformers::core
# ─────────────────────────────────────────────────────────────

add_library(transformers_core INTERFACE)
add_library(transformers::core ALIAS transformers_core)
target_include_directories(transformers_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ─────────────────────────────────────────────────────────────
#  transformers::parser
# ─────────────────────────────────────────────────────────────

add_library(transformers_parser STATIC
    src/parser/safetensors.cpp
    src/parser/config.cpp
    src/parser/weight_map.cpp
)
add_library(transformers::parser ALIAS transformers_parser)
target_include_directories(transformers_parser
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(transformers_parser
    PUBLIC transformers::core
    PRIVATE nlohmann_json::nlohmann_json
)

# ─────────────────────────────────────────────────────────────
#  transformers::resolver
# ─────────────────────────────────────────────────────────────

add_library(transformers_resolver STATIC
    src/resolver/resolver.cpp
)
add_library(transformers::resolver ALIAS transformers_resolver)
target_include_directories(transformers_resolver
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(transformers_resolver
    PRIVATE
        CURL::libcurl
        nlohmann_json::nlohmann_json
)

# ─────────────────────────────────────────────────────────────
#  transformers::backend  (CUDA only for now)
# ─────────────────────────────────────────────────────────────

add_library(transformers_backend STATIC
    src/backend/cuda/device.cu
    src/backend/cuda/tensor.cu
    src/backend/cuda/ops.cu
)
add_library(transformers::backend ALIAS transformers_backend)
target_include_directories(transformers_backend
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(transformers_backend
    PUBLIC transformers::core
    PUBLIC CUDA::cublas
    PUBLIC CUDA::cudart
)
set_target_properties(transformers_backend PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES          "75;80;86;89;90"
)

# ─────────────────────────────────────────────────────────────
#  transformers::models
# ─────────────────────────────────────────────────────────────

add_library(transformers_models STATIC
    src/models/kv_cache.cpp
    src/models/llama/llama_config.cpp
    src/models/llama/llama_model.cpp
    src/models/qwen/qwen_model.cpp
    src/models/gemma/gemma_model.cpp
    src/models/mistral/mistral_model.cpp
)
add_library(transformers::models ALIAS transformers_models)
target_include_directories(transformers_models
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(transformers_models
    PUBLIC transformers::core
    PUBLIC transformers::parser
    PUBLIC transformers::backend
    PRIVATE nlohmann_json::nlohmann_json
)

# ─────────────────────────────────────────────────────────────
#  transformers::tokenizer
# ─────────────────────────────────────────────────────────────

add_library(transformers_tokenizer STATIC
    src/tokenizer/tokenizer.cpp
)
add_library(transformers::tokenizer ALIAS transformers_tokenizer)
target_include_directories(transformers_tokenizer
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(transformers_tokenizer
    PUBLIC  transformers::core
    PRIVATE nlohmann_json::nlohmann_json
)

# ─────────────────────────────────────────────────────────────
#  transformers::inference
# ─────────────────────────────────────────────────────────────

add_library(transformers_inference STATIC
    src/inference/generator.cpp
    src/inference/sampling/greedy.cpp
    src/inference/sampling/top_p.cpp
)
add_library(transformers::inference ALIAS transformers_inference)
target_include_directories(transformers_inference
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(transformers_inference
    PUBLIC transformers::core
    PUBLIC transformers::models
    PUBLIC transformers::tokenizer
    PUBLIC transformers::backend
)

# ─────────────────────────────────────────────────────────────
#  Tools
# ─────────────────────────────────────────────────────────────

option(TRANSFORMERS_BUILD_TOOLS "Build developer tools" OFF)
if(TRANSFORMERS_BUILD_TOOLS)
    add_executable(parser_safetensors tests/parser_safetensors.cpp)
    target_link_libraries(parser_safetensors PRIVATE transformers::parser)

    add_executable(model-cli tools/model-cli.cpp)
    target_link_libraries(model-cli PRIVATE transformers::resolver)

    add_executable(run_llama tools/run_llama.cpp)
    target_link_libraries(run_llama PRIVATE
        transformers::parser
        transformers::backend
        transformers::models
        transformers::tokenizer
        transformers::inference
    )

    add_executable(run_qwen tools/run_qwen.cpp)
    target_link_libraries(run_qwen PRIVATE
        transformers::parser
        transformers::backend
        transformers::models
        transformers::tokenizer
        transformers::inference
    )

    add_executable(run_gemma tools/run_gemma.cpp)
    target_link_libraries(run_gemma PRIVATE
        transformers::parser
        transformers::backend
        transformers::models
        transformers::tokenizer
        transformers::inference
    )
    
    add_executable(run_mistral tools/run_mistral.cpp)
    target_link_libraries(run_mistral PRIVATE
        transformers::parser
        transformers::backend
        transformers::models
        transformers::tokenizer
        transformers::inference
    )
endif()