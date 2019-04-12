cmake_minimum_required(VERSION 3.13.1 FATAL_ERROR)

##############################################
#Compute Platforms

add_subdirectory("CPU")

if(BUILD_CUDA_COMPUTE)

    add_subdirectory("Mock")

endif()

if(BUILD_CUDA_COMPUTE)

    add_subdirectory("CUDA")

endif()


if(BUILD_CUDA_COMPUTE)

    add_subdirectory("OpenCL")

endif()
