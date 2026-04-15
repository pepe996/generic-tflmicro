cmake_minimum_required(VERSION 3.0)

project(tflmicro-test)
set(CXXFLAGS "-std=c++17 -O -I/usr/include/c++/8.3.0/riscv64-poky-linux")
set(CMAKE_CXX_FLAGS "${CXXFLAGS}")
set(DIR_TFLMICRO "${CMAKE_CURRENT_LIST_DIR}/src")
add_subdirectory(${DIR_TFLMICRO})
add_subdirectory(tflm_project/tflm)
