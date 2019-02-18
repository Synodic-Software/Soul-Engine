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

	#TODO: CUDA is not supported in VS2019 Preview
	#enable_language(CUDA)

else()

	message(STATUS "CUDA is missing or a compiler with CUDA support can't be found.")

endif()

#Add CMake tools path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../Tools/CMake")


##############################################
#Sources

file(GLOB_RECURSE PROJECT_HEADERS *.h)
file(GLOB_RECURSE PROJECT_SOURCES *.cpp)
set(PROJECT_FILES 
    ${PROJECT_HEADERS} 
    ${PROJECT_SOURCES} 
)


##############################################
#Dependencies

#Conan via CMake

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../Build/conanbuildinfo_multi.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../Build/conanbuildinfo_multi.cmake)
else()
    include(${CMAKE_CURRENT_SOURCE_DIR}/../Build/conanbuildinfo.cmake)
endif()

conan_basic_setup(TARGETS)

#non-conan dependancies
find_package(Vulkan)


##############################################
#Targets

add_library(SoulEngine STATIC ${PROJECT_FILES})
add_library(synodic::SoulEngine ALIAS SoulEngine)

set_target_properties(SoulEngine
	PROPERTIES 
        LINKER_LANGUAGE CXX
	   	CXX_EXTENSIONS OFF	
		CUDA_SEPARABLE_COMPILATION ON
        CXX_STANDARD 17
)

target_include_directories(SoulEngine
    PUBLIC 
        $<INSTALL_INTERFACE:Include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../Include>

    PRIVATE
        ${CUDA_INCLUDE_DIRS}
		${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(SoulEngine
    PUBLIC
        CONAN_PKG::glm

    INTERFACE

    PRIVATE	
		CONAN_PKG::glfw
        CONAN_PKG::boost
		CONAN_PKG::stb
		Vulkan::Vulkan
)


##############################################
# Installation 

#TODO Does not export headers properly

#Info
install(TARGETS SoulEngine EXPORT SoulEngineTargets
    LIBRARY DESTINATION Libraries
    ARCHIVE DESTINATION Libraries
    RUNTIME DESTINATION Binaries
    INCLUDES DESTINATION Include
)


install(FILES Core/Interface/Application/SoulApplication.h
    DESTINATION Include
    COMPONENT Devel
)

#Export Info
include(CMakePackageConfigHelpers)

set(ConfigPackageSource ${CMAKE_CURRENT_BINARY_DIR}/SoulEngine)
set(ConfigPackageDestination "Libraries/CMake/SoulEngine")

write_basic_package_version_file(
        ${ConfigPackageSource}/SoulEngine-config-version.cmake
        COMPATIBILITY SameMajorVersion)
configure_package_config_file(../Tools/CMake/package-config.cmake.in
        ${ConfigPackageSource}/SoulEngine-config.cmake
        INSTALL_DESTINATION ${ConfigPackageDestination})

#Install
install(EXPORT SoulEngineTargets
        DESTINATION ${ConfigPackageDestination}
        FILE SoulEngineTargets.cmake
        NAMESPACE synodic::
        COMPONENT Devel)
install(FILES
        "${ConfigPackageSource}/SoulEngine-config.cmake"
        "${ConfigPackageSource}/SoulEngine-config-version.cmake"
        DESTINATION ${ConfigPackageDestination}
        COMPONENT Devel)
