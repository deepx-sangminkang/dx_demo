cmake_minimum_required(VERSION 3.16)
project(postprocess_libraries DESCRIPTION "DEEPX Postprocess Libraries")

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

add_subdirectory(yolov11_seg)
