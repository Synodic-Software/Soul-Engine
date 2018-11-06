cmake_minimum_required(VERSION 3.11.1 FATAL_ERROR)


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
include(${CMAKE_CURRENT_SOURCE_DIR}/../Tools/CMake/conan.cmake)
conan_cmake_run(CONANFILE ../conanfile.py
                BASIC_SETUP CMAKE_TARGETS)

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
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:Include>
    PRIVATE
		${CUDA_INCLUDE_DIRS}
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


##############################################
# Installation

#Info
install(TARGETS ${PROJECT_NAME} EXPORT ${CMAKE_PROJECT_NAME}Targets
    LIBRARY DESTINATION Libraries
    ARCHIVE DESTINATION Libraries
    RUNTIME DESTINATION Binaries
    INCLUDES DESTINATION Include
)
install(DIRECTORY #TODO
    DESTINATION Include
    COMPONENT Devel
)

#Export Info
include(CMakePackageConfigHelpers)

set(ConfigPackageSource ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME})
set(ConfigPackageDestination Libraries/CMake/${CMAKE_PROJECT_NAME})
write_basic_package_version_file(
        ${ConfigPackageSource}/${CMAKE_PROJECT_NAME}-config-version.cmake
        COMPATIBILITY SameMajorVersion)
configure_package_config_file(../Tools/CMake/package-config.cmake.in
        ${ConfigPackageSource}/${CMAKE_PROJECT_NAME}-config.cmake
        INSTALL_DESTINATION ${ConfigPackageDestination})

#Install
install(EXPORT ${CMAKE_PROJECT_NAME}Targets
        DESTINATION ${ConfigPackageDestination}
        FILE ${CMAKE_PROJECT_NAME}-targets.cmake
        NAMESPACE synodic::
        COMPONENT Devel)
install(FILES
        "${ConfigPackageSource}/${CMAKE_PROJECT_NAME}-config.cmake"
        "${ConfigPackageSource}/${CMAKE_PROJECT_NAME}-config-version.cmake"
        DESTINATION ${ConfigPackageDestination}
        COMPONENT Devel)
