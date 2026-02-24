cmake_minimum_required(VERSION 3.12)
project(YLLang LANGUAGES CXX CUDA)

# Enable MSVC hot reload if policy supports it
if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
            "$<IF:$<$AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif ()

# Set C++ standard globally to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# Enable generation of compile_commands.json (for clang-tidy)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find clang-format and clang-tidy executables
find_program(CLANG_FORMAT NAMES clang-format)
find_program(CLANG_TIDY NAMES clang-tidy)

# Collect source files to be checked (adjust according to your actual directory structure)
# Recursively search common C/C++ extensions in the current source directory
file(GLOB_RECURSE ALL_CODE_FILES
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hxx"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
)

# ----------------------------------------------------------------------
# Target: clang-format-check
# Run clang-format in dry-run mode, output places where formatting does not conform
# Note: clang-format returns 0 even if issues are found, but output can be examined manually
# ----------------------------------------------------------------------
if (CLANG_FORMAT)
    add_custom_target(clang-format-fix
            COMMAND ${CLANG_FORMAT} -style=file -i ${ALL_CODE_FILES}
            COMMENT "Running clang-format to fix formatting in src/"
            VERBATIM
    )

    add_custom_target(clang-format-check
            COMMAND ${CLANG_FORMAT} -style=file -n ${ALL_CODE_FILES}
            COMMENT "Running clang-format check (dry-run mode)"
            VERBATIM
    )
else ()
    message(WARNING "clang-format not found. 'clang-format-check' target will echo an error.")
    add_custom_target(clang-format-check
            COMMAND ${CMAKE_COMMAND} -E echo "clang-format not available - cannot run check"
            COMMENT "clang-format missing"
    )
endif ()

# ----------------------------------------------------------------------
# Target: clang-tidy-check
# Run clang-tidy for static analysis of the code, using the generated compile_commands.json
# If issues are found, clang-tidy returns a non-zero value, causing the target to fail
# ----------------------------------------------------------------------
if (CLANG_TIDY)
    add_custom_target(clang-tidy-check
            COMMAND ${CLANG_TIDY} -p ${CMAKE_BINARY_DIR} ${ALL_CODE_FILES}
            COMMENT "Running clang-tidy check"
            VERBATIM
    )
else ()
    message(WARNING "clang-tidy not found. 'clang-tidy-check' target will echo an error.")
    add_custom_target(clang-tidy-check
            COMMAND ${CMAKE_COMMAND} -E echo "clang-tidy not available - cannot run check"
            COMMENT "clang-tidy missing"
    )
endif ()


# ----------------------------------------------------------------------
# Target: resolve_config
# Generate C++ constexpr config header from YAML configuration.
# Calls scripts/resolve_config.py to parse config.yaml and produce
# src/include/config/config.h.
# The header is generated at configure time (so it exists for subsequent
# builds). A custom target is also provided for manual regeneration.
# ----------------------------------------------------------------------

# Find Python interpreter (required)
find_package(Python3 REQUIRED)
# Paths to script, input YAML, and output header
set(RESOLVE_CONFIG_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/scripts/resolve_config.py)
set(RESOLVE_CONFIG_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/config/config.yaml)
set(RESOLVE_CONFIG_OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/include/config/config.h)

# Ensure output directory exists (CMake configure-time)
get_filename_component(RESOLVE_CONFIG_OUTPUT_DIR ${RESOLVE_CONFIG_OUTPUT} DIRECTORY)
file(MAKE_DIRECTORY ${RESOLVE_CONFIG_OUTPUT_DIR})

# --- Configure-time generation ---
# Run the script immediately during cmake configuration so that config.h exists
# for the rest of the configure process and for the build.
message(STATUS "Generating config.h with command: ${Python3_EXECUTABLE} ${RESOLVE_CONFIG_SCRIPT} ${RESOLVE_CONFIG_INPUT} ${RESOLVE_CONFIG_OUTPUT}")
execute_process(
        COMMAND ${Python3_EXECUTABLE} ${RESOLVE_CONFIG_SCRIPT} ${RESOLVE_CONFIG_INPUT} ${RESOLVE_CONFIG_OUTPUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE CONFIG_GEN_RESULT
)
message("CONFIG_GEN_RESULT:${CONFIG_GEN_RESULT}")
if (NOT CONFIG_GEN_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to generate config.h from config.yaml. Check your Python script and YAML file.")
endif ()

# Custom command to generate the header
add_custom_command(
        OUTPUT ${RESOLVE_CONFIG_OUTPUT}
        COMMAND Python3::Interpreter ${RESOLVE_CONFIG_SCRIPT} ${RESOLVE_CONFIG_INPUT} ${RESOLVE_CONFIG_OUTPUT}
        DEPENDS ${RESOLVE_CONFIG_SCRIPT} ${RESOLVE_CONFIG_INPUT}
        COMMENT "Generating config.h from config.yaml"
)

# Custom target that depends on the generated file, marked ALL so it runs on every build
add_custom_target(resolve_config ALL
        DEPENDS ${RESOLVE_CONFIG_OUTPUT}
)



# ----------------------------------------------------------------------
# Setup: LibTorch (PyTorch C++ API)
# Sets a cache variable TORCH_INSTALL_DIR for the user to specify the
# installation path of LibTorch. Then finds the Torch package using
# find_package, disabling default search paths to only use the given one.
# Finally, adds the necessary include directories for LibTorch headers
# and the project's own src/include directory.
# ----------------------------------------------------------------------
set(TORCH_INSTALL_DIR "D:/libtorch" CACHE PATH "Path to the LibTorch installation (e.g., where lib/ and include/ are located)")
find_package(Torch REQUIRED PATHS ${TORCH_INSTALL_DIR} NO_DEFAULT_PATH)
include_directories(
        ${TORCH_INSTALL_DIR}/include
        ${TORCH_INSTALL_DIR}/include/torch/csrc/api/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)
add_subdirectory(3rdparty)
list(APPEND CMAKE_PREFIX_PATH ${TVM_FFI_DIR})
message("${TVM_FFI_CMAKE_DIR}")
find_package(tvm_ffi CONFIG REQUIRED)

file(GLOB_RECURSE YLLang_SOURCE_FILES
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hxx"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
)
# Define the executable
add_executable(YLLang ${YLLang_SOURCE_FILES})

# Link Torch libraries (using imported targets automatically handles include/link directories)
target_link_libraries(YLLang PRIVATE ${TORCH_LIBRARIES})

tvm_ffi_configure_target(
        YLLang
)

# On Windows/MSVC, copy Torch DLLs to the output directory after build
file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
add_custom_command(TARGET YLLang POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:YLLang>
        COMMENT "Copying Torch DLLs to output directory"
)
