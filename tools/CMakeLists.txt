# ----------------------------------------------------------------------
# Code formatting and static analysis tools configuration
# ----------------------------------------------------------------------

# Find Python interpreter (required for running helper scripts)
find_package(Python REQUIRED)
set(NATIVE_SOURCE_DIR ${CMAKE_SOURCE_DIR})
# Directory containing helper scripts (run_clang_format.py, run_clang_tidy.py)
set(FORMAT_CHECK_TOOLS_SCRIPTS_DIR "${NATIVE_SOURCE_DIR}/tools")

# ----------------------------------------------------------------------
# clang-format
# ----------------------------------------------------------------------
find_program(CLANG_FORMAT_BIN NAMES clang-format)
if (NOT CLANG_FORMAT_BIN)
    message(WARNING "clang-format not found. Formatting targets will not be available.")
else ()
    message(STATUS "Found clang-format: ${CLANG_FORMAT_BIN}")

    # List of source directories to format (relative to NATIVE_SOURCE_DIR)
    set(FORMAT_SOURCE_DIRS
            "${NATIVE_SOURCE_DIR}/src"
            "${NATIVE_SOURCE_DIR}/test"
    )
    # Join directories with commas as expected by the Python script
    string(JOIN "," FORMAT_DIRS_STR ${FORMAT_SOURCE_DIRS})

    # Target to automatically fix formatting (modifies files in-place)
    add_custom_target(check-format-fix
            COMMAND ${Python_EXECUTABLE} ${FORMAT_CHECK_TOOLS_SCRIPTS_DIR}/run_clang_format.py
            ${CLANG_FORMAT_BIN}
            ${FORMAT_CHECK_TOOLS_SCRIPTS_DIR}/clang_format_exclusions.txt
            --source_dirs ${FORMAT_DIRS_STR}
            --fix
            --quiet
            COMMENT "Running clang-format to fix code style (in-place)"
            VERBATIM
    )

    # Target to check formatting without modifying (dry-run)
    add_custom_target(check-format
            COMMAND ${Python_EXECUTABLE} ${FORMAT_CHECK_TOOLS_SCRIPTS_DIR}/run_clang_format.py
            ${CLANG_FORMAT_BIN}
            ${FORMAT_CHECK_TOOLS_SCRIPTS_DIR}/clang_format_exclusions.txt
            --source_dirs ${FORMAT_DIRS_STR}
            --quiet
            COMMENT "Checking code formatting with clang-format (dry-run)"
            VERBATIM
    )
endif ()

# ----------------------------------------------------------------------
# clang-tidy
# ----------------------------------------------------------------------
find_program(CLANG_TIDY_BIN NAMES clang-tidy)
if (NOT CLANG_TIDY_BIN)
    message(WARNING "clang-tidy not found. Static analysis target will not be available.")
else ()
    message(STATUS "Found clang-tidy: ${CLANG_TIDY_BIN}")

    # Enable generation of compile_commands.json for clang-tidy
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    message("            -clang-tidy-binary ${CLANG_TIDY_BIN} -p ${CMAKE_BINARY_DIR} -header-filter ${NATIVE_SOURCE_DIR}/src/include/.*/.(h|cuh)$ -config-file ${NATIVE_SOURCE_DIR}/.clang-tidy files ${NATIVE_SOURCE_DIR}/src/.*cpp$ ${NATIVE_SOURCE_DIR}/main.cpp ${NATIVE_SOURCE_DIR}/test/.*")
    # 打印当前 CMAKE_CUDA_ARCHITECTURES 的值
    add_custom_target(check-clang-tidy
            COMMAND ${Python_EXECUTABLE} ${FORMAT_CHECK_TOOLS_SCRIPTS_DIR}/run_clang_tidy.py
            -clang-tidy-binary ${CLANG_TIDY_BIN}
            -p ${CMAKE_BINARY_DIR}
            -header-filter "${NATIVE_SOURCE_DIR}/src/include/.*/.(h|cuh)$"
            -config-file "${NATIVE_SOURCE_DIR}/.clang-tidy"
            "${NATIVE_SOURCE_DIR}/src/.*cpp$" "${NATIVE_SOURCE_DIR}/main.cpp" "${NATIVE_SOURCE_DIR}/test/.*"
            COMMENT "Running clang-tidy static analysis"
            VERBATIM
    )
endif ()

# Optional: add a convenience target that runs both checks
add_custom_target(check-all
        COMMENT "Run all code quality checks (format & tidy)"
)
if (TARGET check-format)
    add_dependencies(check-all check-format)
endif ()
if (TARGET check-clang-tidy)
    add_dependencies(check-all check-clang-tidy)
endif ()
