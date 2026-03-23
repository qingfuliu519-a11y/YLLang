message("=============TEST=============")
message(STATUS "Enable test module.")

set(TEST_SOURCE
        "${CMAKE_SOURCE_DIR}/test/mha_kvcache_test.cpp"
        "${CMAKE_SOURCE_DIR}/test/tokenizer_test.cpp"
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
            ${THIRD_PARTY_INCLUDE_DIRS}
    )

    target_link_libraries(${target}
            PRIVATE
            ${TORCH_LIBRARIES}
            CUDA::cudart
            YLLANG_Lib
            minja
            safetensors_cpp
    )

    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_compile_options(${target} PRIVATE -fexceptions)
endforeach ()