message("=============TEST=============")
message(STATUS "Enable test module.")

file(GLOB_RECURSE TEST_SOURCE LIST_DIRECTORIES false ./*.cpp ./*.cu)

foreach (filename IN LISTS TEST_SOURCE)
    get_filename_component(target ${filename} NAME_WLE)
    message(${target})

    add_executable(${target} ${filename})

    target_include_directories(${target}
            PRIVATE ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(${target}
            PRIVATE
            ${TORCH_LIBRARIES}
            CUDA::cudart
            YLLANG_Lib
    )

    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:${target}>
            COMMENT "Copying Torch DLLs to output directory"
    )
endforeach ()