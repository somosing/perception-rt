cmake_minimum_required(VERSION 3.24)

project(perception_rt_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

set(TENSORRT_INCLUDE_DIR "" CACHE PATH "Directory containing NvInferRuntime.h")
set(TENSORRT_LIBRARY "" CACHE FILEPATH "Path to the TensorRT runtime library")
set(CUDART_INCLUDE_DIR "" CACHE PATH "Directory containing cuda_runtime_api.h")
set(CUDART_LIBRARY "" CACHE FILEPATH "Path to the CUDA runtime library")

foreach(required_path
        TENSORRT_INCLUDE_DIR
        TENSORRT_LIBRARY
        CUDART_INCLUDE_DIR
        CUDART_LIBRARY)
    if(NOT ${required_path})
        message(FATAL_ERROR "${required_path} must be provided")
    endif()
endforeach()

if(NOT EXISTS "${TENSORRT_INCLUDE_DIR}/NvInferRuntime.h")
    message(FATAL_ERROR "NvInferRuntime.h was not found in TENSORRT_INCLUDE_DIR")
endif()

if(NOT EXISTS "${CUDART_INCLUDE_DIR}/cuda_runtime_api.h")
    message(FATAL_ERROR "cuda_runtime_api.h was not found in CUDART_INCLUDE_DIR")
endif()

if(NOT EXISTS "${TENSORRT_LIBRARY}")
    message(FATAL_ERROR "TENSORRT_LIBRARY does not exist")
endif()

if(NOT EXISTS "${CUDART_LIBRARY}")
    message(FATAL_ERROR "CUDART_LIBRARY does not exist")
endif()

add_executable(
    perception_rt_native
    src/perception_rt_native.cpp
)

target_include_directories(
    perception_rt_native
    SYSTEM PRIVATE
    "${TENSORRT_INCLUDE_DIR}"
    "${CUDART_INCLUDE_DIR}"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    perception_rt_native
    PRIVATE
    "${TENSORRT_LIBRARY}"
    "${CUDART_LIBRARY}"
    ${OpenCV_LIBS}
    ${CMAKE_DL_LIBS}
)

target_compile_options(
    perception_rt_native
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

get_filename_component(TENSORRT_LIBRARY_DIR "${TENSORRT_LIBRARY}" DIRECTORY)
get_filename_component(CUDART_LIBRARY_DIR "${CUDART_LIBRARY}" DIRECTORY)

set_target_properties(
    perception_rt_native
    PROPERTIES
    BUILD_RPATH "${TENSORRT_LIBRARY_DIR};${CUDART_LIBRARY_DIR}"
)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME native_cli_help COMMAND perception_rt_native --help)
    set_tests_properties(
        native_cli_help
        PROPERTIES
        PASS_REGULAR_EXPRESSION "--image PATH"
    )

    add_test(
        NAME native_cli_rejects_dual_input
        COMMAND perception_rt_native
            --input dummy.fp16
            --image dummy.jpg
    )
    set_tests_properties(
        native_cli_rejects_dual_input
        PROPERTIES
        WILL_FAIL TRUE
    )
endif()
