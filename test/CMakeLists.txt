message("=============TEST=============")
message(STATUS "Enable test module.")

set(TEST_SOURCE
        "${CMAKE_SOURCE_DIR}/test/mha_kvcache_test.cpp"
)

foreach (filename IN LISTS TEST_SOURCE)
    get_filename_component(target ${filename} NAME_WLE)
    message(${target})

    add_executable(${target} ${filename})


    target_include_directories(
            ${target}
            PRIVATE
            ${TORCH_INSTALL_DIR}/include
            ${TORCH_INSTALL_DIR}/include/torch/csrc/api/include
            ${CMAKE_SOURCE_DIR}/src/include
    )

    target_link_libraries(${target}
            PRIVATE
            ${TORCH_LIBRARIES}
            CUDA::cudart
            YLLANG_Lib
    )

    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach ()