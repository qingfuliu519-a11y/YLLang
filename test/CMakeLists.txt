message("=============TEST=============")
message(STATUS "Enable test module.")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

enable_testing()

include(GoogleTest)


file(GLOB_RECURSE TEST_SOURCE LIST_DIRECTORIES false ./*.cpp)

foreach (filename IN LISTS TEST_SOURCE)
    get_filename_component(target ${filename} NAME_WLE)
    message(${target})

    add_executable(${target} ${filename})

    target_include_directories(${target}
            PRIVATE ${CMAKE_SOURCE_DIR}/src
            PUBLIC "D:/vs2022/VC/Tools/MSVC/14.44.35207/include"
    )

    target_link_libraries(${target}
            ${TORCH_LIBRARIES}
            CUDA::cudart
            YLLANG_Lib
            gtest_main
    )

    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    gtest_discover_tests(${target})
endforeach ()

