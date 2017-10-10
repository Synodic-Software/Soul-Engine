cmake_minimum_required(VERSION 3.6.2)
project(Soul_Engine)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(Libraries/glfw-3.2.1)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

find_package(glm 0.9.8 REQUIRED)
include_directories(${GLM_INCLUDE_DIRS} "Include Files")

file(GLOB_RECURSE "SOURCES" "Source Files/*.cpp")

add_executable(Soul_Engine ${SOURCES})
target_link_libraries(Soul_Engine glfw)
