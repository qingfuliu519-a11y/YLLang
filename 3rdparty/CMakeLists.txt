set(THIRD_PARTY_INCLUDE_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/flashinfer/include"
        PARENT_SCOPE)

add_subdirectory(tokenizers-cpp tokenizers EXCLUDE_FROM_ALL)

set(TOKENZIER_CPP_INCLUDE_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/tokenizers-cpp/include"
        PARENT_SCOPE)

