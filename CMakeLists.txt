#pragma once
#include <opencv2/core.hpp>  // 包含 cv::Scalar 和其他基本類型

cmake_minimum_required(VERSION 3.14)

# ==================== Choose Engine（TENSORRT/TFLITE）====================
set(ENGINE "TENSORRT" CACHE STRING "Choose engine: TENSORRT or TFLITE")
set_property(CACHE ENGINE PROPERTY STRINGS "TENSORRT" "TFLITE")

if(ENGINE STREQUAL "TFLITE")
    project(ADAS C CXX)
endif()
if(ENGINE STREQUAL "TENSORRT")
    project(ADAS LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
set(CMAKE_BUILD_TYPE Release)

# ========================= CUDA and TensorRT set =========================
if(ENGINE STREQUAL "TENSORRT")
    # follow your TensorRT CMake: set architectures etc.
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    include(Function)

    set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86 89 90)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

    # Set include/lib Path
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)

    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

    find_package(TensorRT REQUIRED)
    find_package(CUDA REQUIRED)
    
    if (TensorRT_VERSION_MAJOR GREATER_EQUAL 10)
        message(STATUS "Build with -DTRT_10")
        add_definitions(-DTRT_10)
    endif ()

    add_definitions(-DUSE_TENSORRT)
endif()

# ====================== Tensorflow set ======================
if(ENGINE STREQUAL "TFLITE")
    set(TENSORFLOW_SOURCE_DIR "" CACHE PATH "Directory that contains the TensorFlow project")

    if(NOT TENSORFLOW_SOURCE_DIR)
    get_filename_component(TENSORFLOW_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../" ABSOLUTE)
    endif()

    add_subdirectory("${TENSORFLOW_SOURCE_DIR}/tensorflow/lite" "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

    add_definitions(-DUSE_TFLITE)
endif()

# =========================== Src ===========================
aux_source_directory(./src/ SRC_LIST)
aux_source_directory(./src/SORT/ SORT_SRC_LIST)
aux_source_directory(./src/LKA/ LKA_SRC_LIST)

if(ENGINE STREQUAL "TFLITE")
    aux_source_directory(./Engine/TFlite/src/ Engine_SRC_LIST)
elseif(ENGINE STREQUAL "TENSORRT")
    aux_source_directory(./Engine/TensorRT/src/ Engine_SRC_LIST)
endif()

add_executable(${PROJECT_NAME} 
${SRC_LIST}
${SORT_SRC_LIST}
${LKA_SRC_LIST}
${Engine_SRC_LIST}

)

# ========================= Include =========================
include_directories(
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/include/SORT
    ${CMAKE_CURRENT_LIST_DIR}/include/LKA
    
)

if(ENGINE STREQUAL "TFLITE")
    include_directories(
        ${CMAKE_CURRENT_LIST_DIR/Engine/TFlite/include}
    )
elseif(ENGINE STREQUAL "TENSORRT")
    include_directories(
        ${CMAKE_CURRENT_LIST_DIR/Engine/TensorRT/include}
    )
endif()

# ========================= OpenCV =========================
find_package(OpenCV REQUIRED)

# ========================= OpenCL =========================
find_package(OpenCL REQUIRED)

# ========================= Threads =========================
find_package(Threads REQUIRED)

# ========================= Build VDK/GLFW =========================
if(ENGINE STREQUAL "TFLITE")

    set(PLATFORM_LIBS VDK)

    target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        OpenCL
        tensorflow-lite
        dl
        pthread
        drm
        GLESv2
        EGL
    
        glfw   
    #    ${PLATFORM_LIBS}

    )

elseif(ENGINE STREQUAL "TENSORRT")

    set(PLATFORM_LIBS glfw)

    target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        OpenCL
        dl
        pthread
        drm
        GLESv2
        EGL
    
        ${PLATFORM_LIBS}
        ${CUDA_LIBRARIES}   # Ensure CUDA libraries are linked
        ${TensorRT_LIBRARIES}  # Link TensorRT libraries
    )
endif()

# ========================= Build =========================

