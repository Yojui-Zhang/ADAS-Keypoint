cmake_minimum_required(VERSION 3.1)

# --- 選擇引擎（TENSORRT 或 TFLITE）---
set(ENGINE "TENSORRT" CACHE STRING "Choose engine: TENSORRT or TFLITE")
set_property(CACHE ENGINE PROPERTY STRINGS "TENSORRT" "TFLITE")

project(ADAS C CXX)   # project name

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
set(CMAKE_BUILD_TYPE Release)

# ========================= CUDA set =========================
if(ENGINE STREQUAL "TENSORRT")
    # follow your TensorRT CMake: set architectures etc.
    set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86 89 90)
    find_package(CUDA REQUIRED)

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

# ====================== TensorRT set ======================
if(ENGINE STREQUAL "TENSORRT")
    find_package(TensorRT REQUIRED)
    if (TensorRT_VERSION_MAJOR GREATER_EQUAL 10)
        message(STATUS "Build with -DTRT_10")
        add_definitions(-DTRT_10)
    endif ()
endif()

# =========================== Src ===========================
aux_source_directory(./src/ SRC_LIST)

add_executable(${PROJECT_NAME} 
${SRC_LIST}
)

# ========================= Include =========================
include_directories(
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

# if(ENGINE STREQUAL "TENSORRT")
#     include_directories(
#         ${CMAKE_CURRENT_LIST_DIR/Engine/TFlite/}
#     )
# elseif(ENGINE STREQUAL "TFLITE")
#     include_directories(
#         ${CMAKE_CURRENT_LIST_DIR/Engine/TensorRT/}
#     )
# endif()

# ========================= OpenCV =========================
find_package(OpenCV REQUIRED)

# ========================= OpenCL =========================
find_package(OpenCL REQUIRED)

# ========================= Threads =========================
find_package(Threads REQUIRED)

# ========================= VDK or GLFW =========================
if(ENGINE STREQUAL "TFLITE")
  set(PLATFORM_LIBS VDK)

elseif(ENGINE STREQUAL "TENSORRT")
  set(PLATFORM_LIBS glfw)

endif()

# ========================= Build =========================
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    OpenCL
    tensorflow-lite
    dl
    pthread
    drm
    GLESv2
    EGL
    
    ${PLATFORM_LIBS}

)
