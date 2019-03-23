cmake_minimum_required(VERSION 3.13.1 FATAL_ERROR)

##############################################
#Project 

project(SoulInfo)


##############################################
#Dependancies


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
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/Build"
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