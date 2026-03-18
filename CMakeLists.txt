cmake_minimum_required(VERSION 3.12)
# Enable generation of compile_commands.json (for clang-tidy)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(YLLang LANGUAGES CXX CUDA)

# Set C++ standard globally to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND CUDA_NVCC_FLAGS -g -G)
endif ()

add_subdirectory(tools)

# ----------------------------------------------------------------------
# Setup: LibTorch (PyTorch C++ API)
# Sets a cache variable TORCH_INSTALL_DIR for the user to specify the
# installation path of LibTorch. Then finds the Torch package using
# find_package, disabling default search paths to only use the given one.
# Finally, adds the necessary include directories for LibTorch headers
# and the project's own src/include directory.
# ----------------------------------------------------------------------
set(TORCH_INSTALL_DIR "" CACHE PATH "Path to the LibTorch installation (e.g., where lib/ and include/ are located)")
if(NOT TORCH_INSTALL_DIR)
    message(FATAL_ERROR "TORCH_INSTALL_DIR must be specified. Please set it via -DTORCH_INSTALL_DIR=<path>")
endif()
message(STATUS "TORCH_INSTALL_DIR: ${TORCH_INSTALL_DIR}")

find_package(Torch REQUIRED PATHS ${TORCH_INSTALL_DIR} NO_DEFAULT_PATH)
include_directories(
        ${TORCH_INSTALL_DIR}/include
        ${TORCH_INSTALL_DIR}/include/torch/csrc/api/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

# ----------------------------------------------------------------------
# Add src subdirectory, which defines the YLLANG_Lib target
# ----------------------------------------------------------------------
add_subdirectory(3rdparty)
add_subdirectory(src)

# Define the executable
add_executable(YLLANG "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
# Link Torch libraries and YLLANG_Lib (the latter is defined in src/)
target_link_libraries(YLLANG PRIVATE ${TORCH_LIBRARIES} CUDA::cudart YLLANG_Lib)

set_target_properties(YLLANG PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_subdirectory(test)