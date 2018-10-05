cmake_minimum_required(VERSION 3.12.0 FATAL_ERROR)


##############################################
#Options


option(BUILD_TESTS "Build tests." OFF)


##############################################
#Project 

mark_as_advanced(HUNTER_ALLOW_SPACES_IN_PATH)
option(HUNTER_ALLOW_SPACES_IN_PATH ON)

mark_as_advanced(HUNTER_ENABLED)
mark_as_advanced(HUNTER_STATUS_PRINT)
mark_as_advanced(HUNTER_STATUS_DEBUG)
mark_as_advanced(HUNTER_TLS_VERIFY)

include(${CMAKE_CURRENT_SOURCE_DIR}/CMake/Hunter/HunterGate.cmake)

HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.23.27.tar.gz"
    SHA1 "70d75a25f895df3da00d8382417e8c2a08c93173"
    FILEPATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake/Hunter/config.cmake
)

project(SoulEngine 
	VERSION 0.0.1
	DESCRIPTION "Real-time simulation and rendering engine."
	LANGUAGES CXX #CUDA temporarily removed CUDA support
)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#set(EXTRA_CUDA_FLAGS " -arch=sm_50")
#set( CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} ${EXTRA_CUDA_FLAGS}" )


##############################################
#Sources


file(GLOB_RECURSE PROJECT_HEADERS Source/*.h Source/*.cuh Include/*.h)
file(GLOB_RECURSE PROJECT_SOURCES Source/*.cpp Source/*.cu)
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
# First check if glfw is already installed
IF( NOT UNIX )
	hunter_add_package(glfw)
ENDIF()
find_package(glfw3 CONFIG REQUIRED)

#GLM target
hunter_add_package(glm)
find_package(glm REQUIRED)

#stb_image target
hunter_add_package(stb)
find_package(stb CONFIG REQUIRED)

#flatbuffers target
hunter_add_package(flatbuffers)
find_package(Flatbuffers CONFIG REQUIRED)

#googletest target
hunter_add_package(GTest)
find_package(GTest CONFIG REQUIRED)

#boost target
hunter_add_package(Boost COMPONENTS context fiber)
find_package(Boost CONFIG REQUIRED context fiber)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

##############################################
#Targets


add_library(${PROJECT_NAME} STATIC ${PROJECT_FILES})
add_library(Soul::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

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
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Source>
        $<INSTALL_INTERFACE:Include>
        $<INSTALL_INTERFACE:Source>
    PRIVATE
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        
    PRIVATE	
        GTest::main	
		flatbuffers::flatbuffers
		glfw
        Boost::context 
        Boost::fiber 
		Vulkan::Vulkan
        glm
        stb::stb
)


##############################################
#Tests


if(BUILD_TESTS)

    add_subdirectory(Tests)

endif()


##############################################
# Installation

set(INSTALL_CONFIGDIR ${CMAKE_CURRENT_SOURCE_DIR}/Libraries/CMake)
set(INSTALL_INCLUDEDIR ${CMAKE_CURRENT_SOURCE_DIR}/Include)
set(INSTALL_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/Source)
set(INSTALL_LIBDIR ${CMAKE_CURRENT_SOURCE_DIR}/Libraries)
set(INSTALL_LIBEXECDIR ${CMAKE_CURRENT_SOURCE_DIR}/Binaries)

set_target_properties( ${PROJECT_NAME} 
    PROPERTIES 
        EXPORT_NAME ${PROJECT_NAME}
)

install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}-targets
    ARCHIVE DESTINATION ${INSTALL_LIBDIR}
    LIBRARY DESTINATION ${INSTALL_LIBDIR}
    RUNTIME DESTINATION ${INSTALL_LIBEXECDIR}
)

install(EXPORT ${PROJECT_NAME}-targets 
    FILE
        SoulTargets.cmake
    NAMESPACE
        Soul::
    DESTINATION 
        ${INSTALL_CONFIGDIR}
)


include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/CMake/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)


##############################################
#Export


export (
    EXPORT 
        ${PROJECT_NAME}-targets
    FILE 
        SoulTargets.cmake
    NAMESPACE 
        Soul::
)

export(PACKAGE ${PROJECT_NAME})
