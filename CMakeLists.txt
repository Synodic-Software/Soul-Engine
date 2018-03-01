cmake_minimum_required(VERSION 3.9.2 FATAL_ERROR)
project(Soul_Engine 
	VERSION 0.0.1
	DESCRIPTION "Real-time simulation and rendering engine."
	LANGUAGES CXX CUDA
)


##############################################
#Global Variables

set(SOUL_LIBRARIES ${PROJECT_SOURCE_DIR}/Libraries)
set(SOUL_INCLUDE ${PROJECT_SOURCE_DIR}/Includes)
set(SOUL_SOURCE ${PROJECT_SOURCE_DIR}/Source)
set(SOUL_BUILD ${PROJECT_SOURCE_DIR}/Build)
set(SOUL_INSTALL ${PROJECT_SOURCE_DIR}/CMake)

set(APPEND CMAKE_CUDA_FLAGS " -arch=sm_50")
#set(CMAKE_SUPPRESS_REGENERATION true) #zero check is not needed/clutter
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

##############################################
# Submodule Update

# Update all submodules. Taken from `https://cliutils.gitlab.io/modern-cmake/chapters/smallinc.html`
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()


#GLFW target
if(NOT EXISTS "${SOUL_LIBRARIES}/glfw/CMakeLists.txt")
    message(FATAL_ERROR "GLFW was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()
#GLM target
if(NOT EXISTS "${SOUL_LIBRARIES}/glm/CMakeLists.txt")
    message(FATAL_ERROR "GLM was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()
#stb_image target
if(NOT EXISTS "${SOUL_LIBRARIES}/stb/stb_image.h")
    message(FATAL_ERROR "STB was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()
#DI target
if(NOT EXISTS "${SOUL_LIBRARIES}/di/CMakeLists.txt")
    message(FATAL_ERROR "DI was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()
#tinyobjloader target
if(NOT EXISTS "${SOUL_LIBRARIES}/tinyobjloader/CMakeLists.txt")
    message(FATAL_ERROR "tinyobjloader was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()
#Units target
if(NOT EXISTS "${SOUL_LIBRARIES}/units/CMakeLists.txt")
	message(FATAL_ERROR "Units was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()
#googletest target
if(NOT EXISTS "${SOUL_LIBRARIES}/googletest/CMakeLists.txt")
	message(FATAL_ERROR "googletest was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()
#flatbuffers target
if(NOT EXISTS "${SOUL_LIBRARIES}/flatbuffers/CMakeLists.txt")
	message(FATAL_ERROR "flatbuffers was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

##############################################
# Create target and set properties

#GLFW target
set(GLFW_BUILD_EXAMPLES OFF CACHE STRING "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE STRING "" FORCE)
set(GLFW_INSTALL        OFF CACHE STRING "" FORCE)
add_subdirectory(${SOUL_LIBRARIES}/glfw)

#GLM target
#header only

#stb_image target
#header only

#DI target
#header only

#flatbuffers target
add_subdirectory(${SOUL_LIBRARIES}/flatbuffers)

#tinyobjloader target
add_subdirectory(${SOUL_LIBRARIES}/tinyobjloader)

#Units target
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${SOUL_LIBRARIES}/units)

#googletest target
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(${SOUL_LIBRARIES}/googletest)


##############################################
#Main target properties

file(GLOB_RECURSE SourceFiles ${SOUL_SOURCE}/*.cu ${SOUL_SOURCE}/*.cpp)

file(GLOB_RECURSE IncludeFiles ${SOUL_SOURCE}/*.h ${SOUL_SOURCE}/*.cuh)

set(SOUL_All_SOURCES ${SourceFiles} ${IncludeFiles})

file(GLOB_RECURSE InterfaceIncludes ${SOUL_INCLUDE}/*.h)

add_library(SoulEngine STATIC ${SOUL_All_SOURCES})

target_compile_features(SoulEngine 
	PUBLIC 
		cxx_std_17
)

set_target_properties(SoulEngine 
	PROPERTIES 
		CXX_EXTENSIONS OFF	
		CUDA_SEPARABLE_COMPILATION ON
)

#Main target linking and stuff
target_include_directories(SoulEngine
    PUBLIC 
        ${SOUL_INCLUDE}
    PRIVATE
		${SOUL_LIBRARIES}/glfw/include
		${SOUL_LIBRARIES}/glm
        ${SOUL_SOURCE}
)

target_link_libraries(SoulEngine
    PUBLIC
    PRIVATE
		gtest
)


##############################################
# Unit Tests



##############################################
# Installation

#install(TARGETS SoulEngine
#    EXPORT SoulEngine-targets
#    LIBRARY DESTINATION ${SOUL_BUILD}
#    ARCHIVE DESTINATION ${SOUL_BUILD}
#)

#set_target_properties(SoulEngine PROPERTIES EXPORT_NAME SoulEngine)

#bring from 
#install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#install(EXPORT SoulEngine-targets
#	FILE
#		SoulEngineTargets.cmake
#	NAMESPACE
#		SoulEngine::
#	DESTINATION
#		${SOUL_BUILD}
#)


##############################################
# Exporting for others

#export(
#	EXPORT 
#		SoulEngine-targets 
#	FILE 
#		${SOUL_BUILD}/SoulEngineTargets.cmake 
#	NAMESPACE 
#		SoulEngine::
#)

#export(PACKAGE SoulEngine)