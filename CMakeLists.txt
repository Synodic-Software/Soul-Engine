cmake_minimum_required(VERSION 3.11.0 FATAL_ERROR)


##############################################
#Options


option(BUILD_TESTS "Build tests." OFF)
set(HUNTER_ALLOW_SPACES_IN_PATH ON)


##############################################
#Project 


include("CMake/Hunter/HunterGate.cmake")

HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.21.0.tar.gz"
    SHA1 "11967f8ed5218914f1f22393ed36a7b0424f3c09"
    FILEPATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake/Hunter/config.cmake
)

project(SoulEngine 
	VERSION 0.0.1
	DESCRIPTION "Real-time simulation and rendering engine."
	LANGUAGES CXX #CUDA temporarily removed CUDA support
)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(DIRECTORY ${PROJECT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

#set(EXTRA_CUDA_FLAGS " -arch=sm_50")
#set( CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} ${EXTRA_CUDA_FLAGS}" )


##############################################
#Sources


file(GLOB_RECURSE PROJECT_HEADERS Source/*.h Include/*.h)
file(GLOB_RECURSE PROJECT_SOURCES Source/*.cpp)
set (PROJECT_FILES 
    ${PROJECT_HEADERS} 
    ${PROJECT_SOURCES} 
)


include(source_groups)
source_groups(${PROJECT_FILES})


##############################################
#Dependencies


#Vulkan target
find_package(Vulkan REQUIRED)

#GLFW target
hunter_add_package(glfw)
find_package(glfw3 CONFIG REQUIRED)

#GLM target
#header only

#stb_image target
#header only

#DI target
#header only

#flatbuffers target
hunter_add_package(flatbuffers)
find_package(flatbuffers CONFIG REQUIRED)

#googletest target
hunter_add_package(GTest)
find_package(GTest CONFIG REQUIRED)

#boost target
hunter_add_package(Boost COMPONENTS fiber)
find_package(Boost CONFIG REQUIRED fiber)


##############################################
#Targets


add_library(${PROJECT_NAME} STATIC ${PROJECT_FILES})

set_target_properties(${PROJECT_NAME} 
	PROPERTIES 
        LINKER_LANGUAGE CXX
	   	CXX_EXTENSIONS OFF	
		CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(${PROJECT_NAME}
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:Include>
    PRIVATE
#		Libraries/glfw/include
		Libraries/glm
		Libraries/stb
		Libraries/tinyobjloader
        Libraries/entt/src
        Libraries/di/include
 #       ${Boost_INCLUDE_DIRS}
		Vulkan::Vulkan
        Source
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        GTest::main
    PRIVATE		
        flatbuffers::flatbuffers
		glfw
        Boost::fiber 
		Vulkan::Vulkan
)


##############################################
#Tests


if(BUILD_TESTS)
 

endif()


##############################################
# Installation


set_target_properties( ${PROJECT_NAME} 
    PROPERTIES 
        EXPORT_NAME ${PROJECT_NAME}
)

install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}-config
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION Binaries
)

install(
    DIRECTORY
        Include/
    DESTINATION 
        Include
)

install(
    EXPORT  
        ${PROJECT_NAME}-config 
    DESTINATION 
        CMake
    NAMESPACE
        SoulEngine::
)


##############################################
#Export


export (
    EXPORT 
        ${PROJECT_NAME}-config
    FILE 
        ${PROJECT_NAME}-config.cmake
    NAMESPACE 
       SoulEngine::
)

export(PACKAGE ${PROJECT_NAME})