# CMakeList.txt : CMake project for CudaTest, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.9)

project ("CudaTest" LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

file(GLOB_RECURSE src *.cpp)
file(GLOB_RECURSE src_cu *.cu)

add_library(cuda_lib STATIC
	${src_cu}
)

# Add source to this project's executable.
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE cuda_lib)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
	COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/frame.bin" "${CMAKE_CURRENT_BINARY_DIR}/frame.bin"
)	