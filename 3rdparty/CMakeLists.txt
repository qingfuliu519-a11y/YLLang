add_subdirectory(tokenizers-cpp tokenizers EXCLUDE_FROM_ALL)

add_subdirectory(minja)

add_subdirectory(safetensors-cpp)

set(THIRD_PARTY_INCLUDE_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/flashinfer/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/tokenizers-cpp/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/minja/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/safetensors-cpp"
        PARENT_SCOPE)
