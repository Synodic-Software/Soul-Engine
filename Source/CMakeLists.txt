cmake_minimum_required(VERSION 3.13.1 FATAL_ERROR)


##############################################
#Project 

project(SoulEngine 
	VERSION 0.0.1
	DESCRIPTION "Real-time simulation and rendering engine."
	LANGUAGES CXX
)

#CUDA is optional
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)

	enable_language(CUDA)

else()

	message(STATUS "CUDA is missing or a compiler with CUDA support can't be found.")

endif()


#TODO: Implement use of this option
option(BUILD_SHARED_LIBS "Build shared library" OFF)


##############################################
#Sources

#TODO: Get rid of the globs. More elegant solution needed.
file(GLOB_RECURSE PROJECT_HEADERS *.h)
file(GLOB_RECURSE PROJECT_SOURCES *.cpp)
set(PROJECT_FILES 
    ${PROJECT_HEADERS} 
    ${PROJECT_SOURCES} 
)


##############################################
#Targets

add_library(${PROJECT_NAME} ${PROJECT_FILES})
add_library(synodic::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

set_target_properties(${PROJECT_NAME}
	PROPERTIES 
        LINKER_LANGUAGE CXX
	   	CXX_EXTENSIONS OFF	
		CUDA_SEPARABLE_COMPILATION ON
        CXX_STANDARD 17
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/Build"
)


##############################################
#Dependencies

#Conan via CMake
if(EXISTS ${CMAKE_SOURCE_DIR}/Build/conanbuildinfo_multi.cmake)
    #The building case
    include(${CMAKE_SOURCE_DIR}/Build/conanbuildinfo_multi.cmake)
else()
    #The packaging case
    include(${CMAKE_SOURCE_DIR}/Build/conanbuildinfo.cmake)
endif()

conan_basic_setup(TARGETS)

#non-conan dependancies
find_package(Vulkan)

target_include_directories(${PROJECT_NAME}
    PUBLIC 
        $<INSTALL_INTERFACE:Includes>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/Includes>
    PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
		CONAN_PKG::glm

    INTERFACE

    PRIVATE	
		CONAN_PKG::glfw
        CONAN_PKG::boost
		CONAN_PKG::stb
		Vulkan::Vulkan
)