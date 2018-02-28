cmake_minimum_required(VERSION 3.9.2 FATAL_ERROR)
project(Soul_Engine 
	VERSION 0.0.1
	DESCRIPTION "Real-time simulation and rendering engine."
	LANGUAGES CXX CUDA
)


##############################################
#Global Variables

set(LIBRARIES ${CMAKE_SOURCE_DIR}/Libraries)
set(INCLUDE ${CMAKE_SOURCE_DIR}/Includes)
set(SOURCE ${CMAKE_SOURCE_DIR}/Source)
set(BUILD ${CMAKE_SOURCE_DIR}/Build)
set(INSTALL ${CMAKE_SOURCE_DIR}/CMake)

##############################################
# Submodule Update

# Update all submodules. Taken from `https://cliutils.gitlab.io/modern-cmake/chapters/smallinc.html`
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()


##############################################
# Create target and set properties

add_library(SoulEngine ${SOURCE}/Engine\ Core/SoulCore.cpp)

#GLFW target
if(NOT EXISTS "${LIBRARIES}/glfw/CMakeLists.txt")
    message(FATAL_ERROR "GLFW was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

if (NOT TARGET GLFW)
	add_subdirectory(${LIBRARIES}/glfw)
endif ()


#GLM target
if(NOT EXISTS "${LIBRARIES}/glm/CMakeLists.txt")
    message(FATAL_ERROR "GLM was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()


#stb_image target
if(NOT EXISTS "${LIBRARIES}/stb/stb_image.h")
    message(FATAL_ERROR "STB was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()


#DI target
if(NOT EXISTS "${LIBRARIES}/di/CMakeLists.txt")
    message(FATAL_ERROR "DI was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()


#tinyobjloader target
if(NOT EXISTS "${LIBRARIES}/tinyobjloader/CMakeLists.txt")
    message(FATAL_ERROR "tinyobjloader was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

if (NOT TARGET TINYOBJLOADER)
	add_subdirectory(${LIBRARIES}/tinyobjloader)
endif ()


#Units target
if(NOT EXISTS "${LIBRARIES}/units/CMakeLists.txt")
	message(FATAL_ERROR "Units was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

if (NOT TARGET units)
	option(BUILD_TESTS "Remove the BuildTests for units" OFF)
	add_subdirectory(${LIBRARIES}/units)
endif ()


#googletest target
if(NOT EXISTS "${LIBRARIES}/googletest/CMakeLists.txt")
	message(FATAL_ERROR "googletest was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

if (NOT TARGET gtest)
	add_subdirectory(${LIBRARIES}/googletest)
endif ()


#Main target
target_include_directories(SoulEngine
    PUBLIC 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${INCLUDE}>
    PRIVATE
        ${SOURCE}
)

target_compile_features(SoulEngine 
	PUBLIC 
		cxx_std_17
)

set_target_properties(SoulEngine 
	PROPERTIES 
		CXX_EXTENSIONS OFF	
		CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(SoulEngine
    PUBLIC
        
    PRIVATE
)


##############################################
# Unit Tests



##############################################
# Installation

install(TARGETS SoulEngine
    EXPORT SoulEngine-targets
    LIBRARY DESTINATION ${BUILD}
    ARCHIVE DESTINATION ${BUILD}
)

set_target_properties(SoulEngine PROPERTIES EXPORT_NAME SoulEngine)

install(DIRECTORY include/ DESTINATION ${BUILD})

install(EXPORT SoulEngine-targets
	FILE
		SoulEngineTargets.cmake
	NAMESPACE
		SoulEngine::
	DESTINATION
		${BUILD}
)

##############################################
## Exporting for others

#configure_file(${INSTALL}/CMake/FindSoulEngine.cmake
#    ${INSTALL}/FindSoulEngine.cmake
#    COPYONLY)

export(
	EXPORT 
		SoulEngine-targets 
	FILE 
		${INSTALL}/SoulEngineTargets.cmake 
	NAMESPACE 
		SoulEngine::
)

export(PACKAGE SoulEngine)