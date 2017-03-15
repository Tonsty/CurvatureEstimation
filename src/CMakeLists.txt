cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(CurvatureEstimation)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../cmake")

find_package(Eigen3 3.3.0 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(VTK 5.8 REQUIRED)
include(${VTK_USE_FILE})

find_package(FLANN 1.8.4 REQUIRED)
include_directories(${FLANN_INCLUDE_DIR})

file(GLOB CurvatureEstimation_SRC
    "*.h"
    "*.c"
    "*.cpp"
    "*.inl"
)

add_executable(CurvatureEstimation ${CurvatureEstimation_SRC})

target_link_libraries(CurvatureEstimation ${VTK_LIBRARIES} ${FLANN_LIBRARIES})