cmake_minimum_required(VERSION 3.12.1 FATAL_ERROR)

##############################################
#Project 

project(SoulEngineExamples)

#Add CMake tools path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../Tools/CMake")


##############################################
#Dependancies

if(NOT TARGET synodic::SoulEngine)
    find_package(SoulEngine CONFIG REQUIRED)
endif()