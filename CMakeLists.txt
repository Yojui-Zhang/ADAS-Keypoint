cmake_minimum_required(VERSION 3.14)

# ==================== Project Settings ====================
set(ENGINE "TENSORRT" CACHE STRING "Choose engine: TENSORRT or TFLITE")
set_property(CACHE ENGINE PROPERTY STRINGS "TENSORRT" "TFLITE")

if(ENGINE STREQUAL "TFLITE")
    project(ADAS LANGUAGES C CXX)
elseif(ENGINE STREQUAL "TENSORRT")
    project(ADAS LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
set(CMAKE_BUILD_TYPE Release)

# ========================= CUDA and TensorRT Configuration =========================

if(ENGINE STREQUAL "TENSORRT")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    include(Function) # 假設這行是你有自定義函數

    # --------------------------------------------------------
    # [Auto-Detect Path]
    # --------------------------------------------------------
    set(ENV_TRT_ROOT $ENV{TRT_ROOT})
    
    if(DEFINED ENV_TRT_ROOT AND NOT ENV_TRT_ROOT STREQUAL "")
        message(STATUS ">>>> Mode: Custom TensorRT Path (Ubuntu/Desktop)")
        list(APPEND CMAKE_PREFIX_PATH "${ENV_TRT_ROOT}")
        
        # 【關鍵修正】：明確加入 Tarball 的 include 資料夾
        include_directories("${ENV_TRT_ROOT}/include")
        
        # 也可以順便加入 lib 路徑以防萬一
        link_directories("${ENV_TRT_ROOT}/lib")
    else()
        message(STATUS ">>>> Mode: System TensorRT Path (Jetson Orin/Default)")
        # 系統路徑通常在 /usr/include/aarch64-linux-gnu/ 或 /usr/include
        # 為了保險起見，可以加入這個常見路徑 (Orin 上通常不需要，但加了無妨)
        include_directories("/usr/include/aarch64-linux-gnu")
    endif()

    # ... (中間架構判斷保持不變) ...
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(CMAKE_CUDA_ARCHITECTURES 87)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 61 75 86)
    endif()
    
    
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)

    find_package(CUDA REQUIRED)
    find_package(TensorRT REQUIRED)
    
    # 【保險修正】：有些 FindTensorRT.cmake 會將路徑放在這個變數，也加進去
    if(TensorRT_INCLUDE_DIRS)
        include_directories(${TensorRT_INCLUDE_DIRS})
    endif()

    # ... (後續版本定義保持不變) ...
    if (TensorRT_VERSION_MAJOR GREATER_EQUAL 10)
        add_definitions(-DTRT_10)
    else()
        add_definitions(-DTRT_LEGACY)
    endif ()

    add_definitions(-DUSE_TENSORRT)
endif()

# ====================== Tensorflow Lite Set ======================
if(ENGINE STREQUAL "TFLITE")
    set(TENSORFLOW_SOURCE_DIR "" CACHE PATH "Directory that contains the TensorFlow project")
    if(NOT TENSORFLOW_SOURCE_DIR)
        get_filename_component(TENSORFLOW_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../" ABSOLUTE)
    endif()
    add_subdirectory("${TENSORFLOW_SOURCE_DIR}/tensorflow/lite" "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)
    add_definitions(-DUSE_TFLITE)
endif()

# =========================== Source Files ===========================
aux_source_directory(./src/ SRC_LIST)
aux_source_directory(./src/SORT/ SORT_SRC_LIST)
aux_source_directory(./src/LKA/ LKA_SRC_LIST)
aux_source_directory(./src/Geometry/ Geometry_SRC_LIST)


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
    ${Geometry_SRC_LIST}
)

# ========================= Include Directories =========================

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/include/SORT
    ${CMAKE_CURRENT_LIST_DIR}/include/LKA
    ${CMAKE_CURRENT_LIST_DIR}/include/Geometry
)

if(ENGINE STREQUAL "TFLITE")
    include_directories(${CMAKE_CURRENT_LIST_DIR}/Engine/TFlite/include)
elseif(ENGINE STREQUAL "TENSORRT")
    include_directories(${CMAKE_CURRENT_LIST_DIR}/Engine/TensorRT/include)
endif()

# ========================= Libraries Dependencies =========================
find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)

# ========================= Link Libraries =========================
if(ENGINE STREQUAL "TFLITE")
    set(PLATFORM_LIBS VDK) # Check if VDK is actually needed or defined elsewhere
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
        # ${PLATFORM_LIBS}
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
        ${CUDA_LIBRARIES}      # 連結 CUDA
        ${TensorRT_LIBRARIES}  # 連結 TensorRT
    )
endif()