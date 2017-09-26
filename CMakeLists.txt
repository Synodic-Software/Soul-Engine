cmake_minimum_required(VERSION 3.6.2)
project(Soul_Engine)

set(CMAKE_CXX_STANDARD 11)

file(GLOB_RECURSE "INCLUDES" "*.h")
file(GLOB_RECURSE "SOURCES" "*.cpp" "*.h")

add_executable(Soul_Engine ${SOURCES} ${INCLUDES})