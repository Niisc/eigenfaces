#cmake_minimum_required(VERSION 3.26)
project(eigenfaces)

#set(CMAKE_CXX_STANDARD 17)

find_package(ArrayFire)

add_executable(eigenfaces.out main.cpp)

target_link_libraries(eigenfaces.out ArrayFire::afcuda)

