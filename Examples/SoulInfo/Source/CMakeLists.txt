cmake_minimum_required(VERSION 3.13.1 FATAL_ERROR)

##############################################
#Project 

project(SoulInfo)

#Add CMake tools path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../Tools/CMake")


##############################################
#Dependancies

if(NOT TARGET synodic::SoulEngine)
    find_package(SoulEngine CONFIG REQUIRED)
endif()


##############################################
#Sources

file(GLOB_RECURSE PROJECT_HEADERS *.h)
file(GLOB_RECURSE PROJECT_SOURCES *.cpp)
set(PROJECT_FILES 
    ${PROJECT_HEADERS} 
    ${PROJECT_SOURCES} 
)


##############################################
#Targets

add_executable(SoulInfo ${PROJECT_FILES})

set_target_properties(SoulInfo
	PROPERTIES 
        LINKER_LANGUAGE CXX
	   	CXX_EXTENSIONS OFF	
        CXX_STANDARD 17
)

target_include_directories(SoulInfo
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/Source>
    PRIVATE

)

target_link_libraries(SoulInfo
    PUBLIC

    INTERFACE

    PRIVATE	
        synodic::SoulEngine
)