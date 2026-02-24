# ----------------------------------------------------------------------
# Update tvm-ffi git submodule
# Ensures that the tvm-ffi submodule is properly initialized and updated
# recursively. This step fetches the required source code before building.
# ----------------------------------------------------------------------
find_package(Git REQUIRED)
message(STATUS "Updating tvm-ffi git submodule...")
execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE GIT_RESULT
        OUTPUT_VARIABLE GIT_OUTPUT
        ERROR_VARIABLE GIT_ERROR
)
message(STATUS "Git submodule update result: ${GIT_RESULT}")
message(STATUS "tGit submodule update output: ${GIT_OUTPUT}")
message(STATUS "Git submodule update error:  ${GIT_ERROR}")
if (NOT GIT_RESULT EQUAL 0)
    message(FATAL_ERROR "Git submodule update failed with result ${GIT_RESULT}")
endif ()

# ----------------------------------------------------------------------
# Define installation directory for tvm-ffi Python package
# Sets TVM_FFI_INSTALL_DIR to a location within the build tree
# (${CMAKE_BINARY_DIR}/3rdparty/tvm-ffi). This directory will hold the
# installed Python package after pip install.
# ----------------------------------------------------------------------
set(TVM_FFI_INSTALL_DIR "${CMAKE_SOURCE_DIR}/3rdparty/tvm-ffi")
message(STATUS "tvm-ffi will be installed to: ${TVM_FFI_INSTALL_DIR}")

# Ensure the target directory exists
file(MAKE_DIRECTORY ${TVM_FFI_INSTALL_DIR})

# ----------------------------------------------------------------------
# Install tvm-ffi Python package via pip
# Runs pip install (non-editable) to the target directory. The installation
# happens at configure time to ensure the package is available for later
# find_package calls. Any failure aborts the configuration.
# ----------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter)

message(STATUS "Installing tvm-ffi Python package via pip...")
message(STATUS "Python_EXECUTABLE = ${Python_EXECUTABLE}")
message(STATUS "Working directory = ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm-ffi")
message(STATUS "COMMAND = ${Python_EXECUTABLE} -m pip install --target ${TVM_FFI_INSTALL_DIR} --verbose .")
execute_process(
        COMMAND ${Python_EXECUTABLE} -m pip install --target ${TVM_FFI_INSTALL_DIR} --verbose .
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/3rdparty/tvm-ffi
        RESULT_VARIABLE PIP_RESULT
        OUTPUT_VARIABLE PIP_OUTPUT
        ERROR_VARIABLE PIP_ERROR
        COMMAND_ERROR_IS_FATAL ANY   # If command fails, abort configuration
)

message(STATUS "tvm-ffi install result: ${PIP_RESULT}")
message(STATUS "tvm-ffi install output: ${PIP_OUTPUT}")
message(STATUS "tvm-ffi install error:  ${PIP_ERROR}")

if (NOT PIP_RESULT EQUAL 0)
    message(FATAL_ERROR "pip install failed with result ${PIP_RESULT}")
endif ()

# ----------------------------------------------------------------------
# Set CMake cache variable TVM_FFI_DIR
# After installation, points TVM_FFI_DIR to the directory containing
# tvm-ffi's CMake configuration files. This cached variable is globally
# visible, allowing find_package(tvm_ffi) to locate the package even
# from parent scopes. Adjust the path if the actual config file location
# differs (e.g., under lib/cmake/tvm_ffi).
# ----------------------------------------------------------------------
set(TVM_FFI_DIR "${TVM_FFI_INSTALL_DIR}/tvm_ffi" CACHE PATH "Path to tvm-ffi CMake configuration files" FORCE)
message(STATUS "Set TVM_FFI_DIR = ${TVM_FFI_DIR}")

# ----------------------------------------------------------------------
# Extend PYTHONPATH for configure-time Python scripts
# Prepends the tvm-ffi Python package directory to the PYTHONPATH
# environment variable, preserving any existing paths. This ensures that
# any Python scripts executed during configuration (e.g., import tests)
# can locate the newly installed tvm-ffi package. Path separator is
# platform-aware (';' on Windows, ':' elsewhere).
# ----------------------------------------------------------------------
if(WIN32)
    # Windows uses semicolon as path separator
    set(ENV{PYTHONPATH} "${TVM_FFI_INSTALL_DIR};$ENV{PYTHONPATH}")
else()
    # Unix/Linux/macOS use colon as path separator
    set(ENV{PYTHONPATH} "${TVM_FFI_INSTALL_DIR}:$ENV{PYTHONPATH}")
endif()