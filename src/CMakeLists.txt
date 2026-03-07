# src/CMakeLists.txt

# Collect source files (recursively) for the YLLANG library
file(GLOB_RECURSE YLLang_SOURCE_FILES
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh"
)

# Define the library
add_library(YLLANG_Lib ${YLLang_SOURCE_FILES})

# Link Torch libraries and CUDA runtime
target_link_libraries(YLLANG_Lib PRIVATE ${TORCH_LIBRARIES} CUDA::cudart)

# Enable separable compilation for CUDA
set_target_properties(YLLANG_Lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include directories for the library's own headers (src/include)
target_include_directories(YLLANG_Lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

