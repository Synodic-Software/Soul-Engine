cmake_minimum_required(VERSION 3.11.0 FATAL_ERROR)
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

set(EXTRA_CUDA_FLAGS " -arch=sm_50")
set(CUDA_VISIBLES  "5")

set( CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} ${EXTRA_CUDA_FLAGS}" )
set( CUDA_VISIBLE_DEVICES  "${CUDA_VISIBLE_DEVICES} ${CUDA_VISIBLES}" )


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

#entt target
if(NOT EXISTS "${SOUL_LIBRARIES}/entt/CMakeLists.txt")
    message(FATAL_ERROR "entt was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
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

find_package(Vulkan REQUIRED)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
#set(GLFW_VULKAN_STATIC  ON  CACHE BOOL "" FORCE)
add_subdirectory(${SOUL_LIBRARIES}/glfw)

#GLM target
#header only

#stb_image target
#header only

#DI target
#header only

#flatbuffers target
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${SOUL_LIBRARIES}/flatbuffers)

#googletest target
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(${SOUL_LIBRARIES}/googletest)

add_definitions(-DBOOST_ALL_NO_LIB)

set(Boost_USE_STATIC_LIBS        ON)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)

SET(BOOST_ROOT ${PROJECT_SOURCE_DIR}/Libraries/boost)
find_package(Boost 1.66.0 REQUIRED COMPONENTS fiber)


##############################################
#Main target properties

file(GLOB_RECURSE SourceFiles 
	LIST_DIRECTORIES false 
	${SOUL_SOURCE}/*.cu ${SOUL_SOURCE}/*.cpp)

file(GLOB_RECURSE IncludeFiles 
	LIST_DIRECTORIES false 
	${SOUL_SOURCE}/*.h ${SOUL_SOURCE}/*.cuh)

file(GLOB SOUL_All_SOURCES ${SourceFiles} ${IncludeFiles})

#add_library(SoulEngine STATIC ${SOUL_All_SOURCES})
add_executable(SoulEngine ${SOUL_All_SOURCES})
set_property(DIRECTORY ${PROJECT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT SoulEngine)


#Visual filters for all source files
foreach(_source IN ITEMS ${SOUL_All_SOURCES})
    get_filename_component(_source_path "${_source}" PATH)
    file(RELATIVE_PATH _source_path_rel "${PROJECT_SOURCE_DIR}" "${_source_path}")
    string(REPLACE "/" "\\" _group_path "${_source_path_rel}")
    source_group("${_group_path}" FILES "${_source}")
endforeach()

#Visual filters for all frontend files
foreach(_source IN ITEMS ${SOUL_INCLUDE})
    get_filename_component(_source_path "${_source}" PATH)
    file(RELATIVE_PATH _source_path_rel "${PROJECT_SOURCE_DIR}" "${_source_path}")
    string(REPLACE "/" "\\" _group_path "${_source_path_rel}")
    source_group("${_group_path}" FILES "${_source}")
endforeach()


set_property(TARGET SoulEngine PROPERTY CXX_STANDARD 17)
set_property(TARGET SoulEngine PROPERTY CXX_STANDARD_REQUIRED ON)

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
		${SOUL_LIBRARIES}/stb
		${SOUL_LIBRARIES}/tinyobjloader
        ${SOUL_LIBRARIES}/entt/src
        ${SOUL_LIBRARIES}/di/include
		${Boost_INCLUDE_DIRS}
		Vulkan::Vulkan
        ${SOUL_SOURCE}
)

target_link_libraries(SoulEngine
    PUBLIC
  
    PRIVATE		
		glfw ${GLFW_LIBRARIES}
		Boost::boost Boost::fiber 
		Vulkan::Vulkan
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