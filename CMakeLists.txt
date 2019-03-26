cmake_minimum_required(VERSION 3.13.1 FATAL_ERROR)


##############################################
#Project 

project(SoulEngine)

#TODO: Implement testing framework. Once that is done, this option can be defaulted to ON
option(BUILD_TESTS "Build test projects" OFF)
option(BUILD_EXAMPLES "Build example projects" ON)


##############################################
#CMake delegation

add_subdirectory(Source)

if(BUILD_TESTS)

    add_subdirectory(Tests)

endif()

if(BUILD_EXAMPLES)

    add_subdirectory(Examples)

endif()