cmake_minimum_required(VERSION 3.12.1 FATAL_ERROR)


##############################################
#Project 

project(SoulEngine 
	VERSION 0.0.1
	DESCRIPTION "Real-time simulation and rendering engine."
	LANGUAGES CXX
)

#CUDA is optional
enable_language(CUDA)

#Add CMake tools path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../Tools/CMake")

#options
set(EXE_REPOSITORY_PATH "")

##############################################
#Sources

file(GLOB_RECURSE PROJECT_HEADERS *.h *.cuh)
file(GLOB_RECURSE PROJECT_SOURCES *.cpp *.cu)
set (PROJECT_FILES 
    ${PROJECT_HEADERS} 
    ${PROJECT_SOURCES} 
)


##############################################
#Dependencies

#Conan via CMake
include(${CMAKE_SOURCE_DIR}/Build/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

#non-conan dependancies
find_package(Vulkan REQUIRED)
find_package(CUDA)


##############################################
#Targets

add_library(${PROJECT_NAME} STATIC ${PROJECT_FILES})
add_library(synodic::${PROJECT_NAME} ALIAS ${PROJECT_NAME})


set_target_properties(${PROJECT_NAME} 
	PROPERTIES 
        LINKER_LANGUAGE CXX
	   	CXX_EXTENSIONS OFF	
		CUDA_SEPARABLE_COMPILATION ON
        CXX_STANDARD 17
)

target_include_directories(${PROJECT_NAME}
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
      
    INTERFACE

    PRIVATE	
		CONAN_PKG::glfw
        CONAN_PKG::boost
        CONAN_PKG::glm
		CONAN_PKG::stb
		Vulkan::Vulkan
)

if ("${EXE_REPOSITORY_PATH}" STREQUAL "")

    #TODO Pull Conan package for default EXE

else()

    add_subdirectory(${EXE_REPOSITORY_PATH})

endif()

##############################################
# Installation 

#TODO Does not export headers properly

#Info
install(TARGETS ${PROJECT_NAME} EXPORT SoulEngineTargets
    LIBRARY DESTINATION Libraries
    ARCHIVE DESTINATION Libraries
    RUNTIME DESTINATION Binaries
    INCLUDES DESTINATION Include
)

set (PROJECT_INTERFACE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/Interface/Application/SoulApplication.h
)

install(FILES ${PROJECT_INTERFACE_HEADERS}
    DESTINATION Include
    COMPONENT Devel
)

#Export Info
include(CMakePackageConfigHelpers)

set(ConfigPackageSource ${CMAKE_CURRENT_BINARY_DIR}/SoulEngine)
set(ConfigPackageDestination Libraries/CMake/SoulEngine)

write_basic_package_version_file(
        ${ConfigPackageSource}/SoulEngine-config-version.cmake
        COMPATIBILITY SameMajorVersion)
configure_package_config_file(../Tools/CMake/package-config.cmake.in
        ${ConfigPackageSource}/SoulEngine-config.cmake
        INSTALL_DESTINATION ${ConfigPackageDestination})

#Install
install(EXPORT SoulEngineTargets
        DESTINATION ${ConfigPackageDestination}
        FILE SoulEngine-targets.cmake
        NAMESPACE synodic::
        COMPONENT Devel)
install(FILES
        "${ConfigPackageSource}/SoulEngine-config.cmake"
        "${ConfigPackageSource}/SoulEngine-config-version.cmake"
        DESTINATION ${ConfigPackageDestination}
        COMPONENT Devel)
