add_subdirectory(tokenizers-cpp tokenizers EXCLUDE_FROM_ALL)
add_subdirectory(minja)
set(THIRD_PARTY_INCLUDE_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/flashinfer/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/tokenizers-cpp/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/minja/include"
        PARENT_SCOPE)
