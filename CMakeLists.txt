cmake_minimum_required(VERSION 3.12.1 FATAL_ERROR)


##############################################
#Project 

project(SoulEngine)

#Add CMake tools path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Tools/CMake")


##############################################
#CMake delegation

add_subdirectory(Source)
add_subdirectory(Tests)
add_subdirectory(Examples)