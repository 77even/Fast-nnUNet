cmake_minimum_required(VERSION 3.30)
project(FastnnUNet CUDA CXX)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
#    add_compile_options(/utf-8)
    add_compile_definitions(_UNICODE UNICODE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8 /O3 /arch:AVX2 /favor:INTEL64")

endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


set(PROJECT_DIRS
        utils
)
foreach(DIR ${PROJECT_DIRS})
    include_directories(${DIR}/include)
endforeach()

set(EVALUATION_DIRS
        cuda
        exam
        infer
        utils
)
foreach(DIR ${EVALUATION_DIRS})
    list(APPEND PROJECT_DIRS evaluation/${DIR})
    include_directories(evaluation/${DIR}/include)
endforeach()

set(TensorRT_DIR $ENV{TensorRT_PATH})

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(VTK REQUIRED)
find_package(ITK REQUIRED)
find_package(SimpleITK REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_DIR}/include
        ${VTK_INCLUDE_DIRS}
        ${ITK_INCLUDE_DIRS}
        ${SimpleITK_INCLUDE_DIRS}
)

file(GLOB_RECURSE HEADER_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/evaluation/*/include/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/evaluation/*/include/*.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/utils/include/*.h"
)

file(GLOB_RECURSE SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/evaluation/*/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/evaluation/*/src/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/utils/src/*.cpp"
)

add_executable(FastnnUNet ${HEADER_FILES} ${SOURCE_FILES} main.cpp)

target_link_directories(FastnnUNet PRIVATE
        ${TensorRT_DIR}/lib
)

target_link_libraries(FastnnUNet PRIVATE
        ${OpenCV_LIBS}
        ${VTK_LIBRARIES}
        ${ITK_LIBRARIES}
        ${SimpleITK_LIBRARIES}
        CUDA::cudart
        nvinfer_10
        nvinfer_plugin_10
        nvonnxparser_10
)
