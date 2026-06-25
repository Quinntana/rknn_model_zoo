cmake_minimum_required(VERSION 3.10)

project(face_benchmark)

if (ENABLE_ASAN)
    message(STATUS "BUILD WITH ADDRESS SANITIZER")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${PROJECT_NAME}
    main.cc
)

target_link_libraries(${PROJECT_NAME}
    imageutils
    fileutils
    ${LIBRKNNRT}
    dl
)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME} log)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../RetinaFace/cpp
    ${LIBRKNNRT_INCLUDES}
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/model_registry.yaml DESTINATION model)

set(sample_image ${CMAKE_CURRENT_SOURCE_DIR}/../../RetinaFace/model/test.jpg)
if (EXISTS ${sample_image})
    install(FILES ${sample_image} DESTINATION model)
endif()

file(GLOB SAMPLE_RKNN_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../yolo11/model/*.rknn"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../RetinaFace/model/*.rknn"
)
if (SAMPLE_RKNN_FILES)
    install(FILES ${SAMPLE_RKNN_FILES} DESTINATION model)
endif()
