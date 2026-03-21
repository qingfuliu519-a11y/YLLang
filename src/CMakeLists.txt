# src/CMakeLists.txt

find_package(CURL REQUIRED)

# Collect source files (recursively) for the YLLANG library
set(YLLang_SOURCE_FILES
        "${CMAKE_SOURCE_DIR}/src/layers/base_layer.cpp"
        "${CMAKE_SOURCE_DIR}/src/kvcache/store_kv_cache.cu"
        "${CMAKE_SOURCE_DIR}/src/models/model.cpp"
        "${CMAKE_SOURCE_DIR}/src/models/qwen3.cpp"
        "${CMAKE_SOURCE_DIR}/src/engine/engine.cpp"
        "${CMAKE_SOURCE_DIR}/src/util/curl_downloader.cpp"
)

# Define the library
add_library(YLLANG_Lib ${YLLang_SOURCE_FILES})

# Link Torch libraries and CUDA runtime
target_link_libraries(YLLANG_Lib PRIVATE ${TORCH_LIBRARIES} CUDA::cudart tokenizers_cpp CURL::libcurl minja)

# Enable separable compilation for CUDA
set_target_properties(YLLANG_Lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include directories for the library's own headers (src/include)
target_include_directories(
        YLLANG_Lib
        PRIVATE
        ${TORCH_INSTALL_DIR}/include
        ${TORCH_INSTALL_DIR}/include/torch/csrc/api/include
        ${CMAKE_SOURCE_DIR}/src/include
        ${THIRD_PARTY_INCLUDE_DIRS})