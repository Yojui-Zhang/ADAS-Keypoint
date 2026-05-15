cmake_minimum_required(VERSION 3.14)

set(ENGINE "TENSORRT" CACHE STRING "Choose engine: TENSORRT or TFLITE")
set_property(CACHE ENGINE PROPERTY STRINGS "TENSORRT" "TFLITE")

if(ENGINE STREQUAL "TFLITE")
    project(ADAS LANGUAGES C CXX)
elseif(ENGINE STREQUAL "TENSORRT")
    project(ADAS LANGUAGES C CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
set(CMAKE_BUILD_TYPE Release)

set(CANBUS "ITRI" CACHE STRING "Choose Canbus: ITRI or YUNTECH")
set_property(CACHE CANBUS PROPERTY STRINGS "ITRI" "YUNTECH")

if(ENGINE STREQUAL "TENSORRT")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    include(Function)

    set(ENV_TRT_ROOT $ENV{TRT_ROOT})

    if(DEFINED ENV_TRT_ROOT AND NOT ENV_TRT_ROOT STREQUAL "")
        message(STATUS ">>>> Mode: Custom TensorRT Path (Ubuntu/Desktop)")
        list(APPEND CMAKE_PREFIX_PATH "${ENV_TRT_ROOT}")
        include_directories("${ENV_TRT_ROOT}/include")
        link_directories("${ENV_TRT_ROOT}/lib")
    else()
        message(STATUS ">>>> Mode: System TensorRT Path (Jetson Orin/Default)")
        include_directories("/usr/include/aarch64-linux-gnu")
    endif()

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

    if(TensorRT_INCLUDE_DIRS)
        include_directories(${TensorRT_INCLUDE_DIRS})
    endif()

    if (TensorRT_VERSION_MAJOR GREATER_EQUAL 10)
        add_definitions(-DTRT_10)
    else()
        add_definitions(-DTRT_LEGACY)
    endif ()

    add_definitions(-DUSE_TENSORRT)
endif()

if(ENGINE STREQUAL "TFLITE")
    set(TENSORFLOW_SOURCE_DIR "" CACHE PATH "Directory that contains the TensorFlow project")
    if(NOT TENSORFLOW_SOURCE_DIR)
        get_filename_component(TENSORFLOW_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../" ABSOLUTE)
    endif()
    add_subdirectory("${TENSORFLOW_SOURCE_DIR}/tensorflow/lite" "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)
    add_definitions(-DUSE_TFLITE)
endif()

set(SRC_LIST
    ./src/main.cpp
    ./src/config.cc
    ./src/system_config.cpp
    ./src/imageShow.cpp
)

aux_source_directory(./src/Camera/ CAMERA_SRC_LIST)
aux_source_directory(./src/SORT/ SORT_SRC_LIST)
aux_source_directory(./src/Geometry/ Geometry_SRC_LIST)
aux_source_directory(./src/LKA/ LKA_SRC_LIST)
aux_source_directory(./src/ACC/ ACC_SRC_LIST)
aux_source_directory(./src/StabilityControl/ StabilityControl_SRC_LIST)
aux_source_directory(./src/VehicleBehavior/ VehicleBehavior_SRC_LIST)
aux_source_directory(./src/DRAW/ DRAW_SRC_LIST)
aux_source_directory(./src/Collision/ Collision_SRC_LIST)

set(KEYPAD_SRC_LIST
    ./src/keypad/keypad.cpp
    ./src/keypad/keypad_control.cpp
)

set(LOG_SRC_LIST
    ./src/log/research_data_logger.cpp
    ./src/log/algorithm_ablation_logger.cpp
    ./src/log/time_sync.cpp
    ./src/log/runtime_log_manager.cpp
)

if(ENGINE STREQUAL "TFLITE")
    aux_source_directory(./Engine/TFlite/src/ Engine_SRC_LIST)
elseif(ENGINE STREQUAL "TENSORRT")
    aux_source_directory(./Engine/TensorRT/src/ Engine_SRC_LIST)
endif()

if(CANBUS STREQUAL "ITRI")
    aux_source_directory(./CANBus/itri/src/ CANBUS_SRC_LIST )
    add_definitions(-DUSE_ITRI_CAN)
elseif(CANBUS STREQUAL "YUNTECH")
    aux_source_directory(./CANBus/yuntech/src/ CANBUS_SRC_LIST )
    add_definitions(-DUSE_YUNTECH_CAN)
endif()

add_executable(${PROJECT_NAME}
    ${SRC_LIST}
    ${CAMERA_SRC_LIST}
    ${CANBUS_SRC_LIST}
    ${Engine_SRC_LIST}
    ${SORT_SRC_LIST}
    ${Geometry_SRC_LIST}
    ${LKA_SRC_LIST}
    ${ACC_SRC_LIST}
    ${VehicleBehavior_SRC_LIST}
    ${DRAW_SRC_LIST}
    ${StabilityControl_SRC_LIST}
    ${Collision_SRC_LIST}
    ${KEYPAD_SRC_LIST}
    ${LOG_SRC_LIST}
)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/include/Camera
    ${CMAKE_CURRENT_LIST_DIR}/include/SORT
    ${CMAKE_CURRENT_LIST_DIR}/include/Geometry
    ${CMAKE_CURRENT_LIST_DIR}/include/LKA
    ${CMAKE_CURRENT_LIST_DIR}/include/ACC
    ${CMAKE_CURRENT_LIST_DIR}/include/StabilityControl
    ${CMAKE_CURRENT_LIST_DIR}/include/VehicleBehavior
    ${CMAKE_CURRENT_LIST_DIR}/include/DRAW
    ${CMAKE_CURRENT_LIST_DIR}/include/Collision
    ${CMAKE_CURRENT_LIST_DIR}/include/Keypad
    ${CMAKE_CURRENT_LIST_DIR}/include/log
)

if(ENGINE STREQUAL "TFLITE")
    include_directories(${CMAKE_CURRENT_LIST_DIR}/Engine/TFlite/include)
elseif(ENGINE STREQUAL "TENSORRT")
    include_directories(${CMAKE_CURRENT_LIST_DIR}/Engine/TensorRT/include)
endif()

if(CANBUS STREQUAL "ITRI")
    include_directories(${CMAKE_CURRENT_LIST_DIR}/CANBus/itri/include)
elseif(CANBUS STREQUAL "YUNTECH")
    include_directories(${CMAKE_CURRENT_LIST_DIR}/CANBus/yuntech/include)
endif()

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)

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
        # ${PLATFORM_LIBS}
        glfw
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
        ${CUDA_LIBRARIES}
        ${TensorRT_LIBRARIES}
    )
endif()
