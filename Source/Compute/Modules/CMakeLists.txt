cmake_minimum_required(VERSION 3.13.1 FATAL_ERROR)

##############################################
#Compute Platforms

if(BUILD_CPU_COMPUTE)

    add_subdirectory("CPU")

endif()

if(BUILD_CUDA_COMPUTE)

    add_subdirectory("Mock")

endif()

if(BUILD_CUDA_COMPUTE)

    add_subdirectory("CUDA")

endif()


if(BUILD_CUDA_COMPUTE)

    add_subdirectory("OpenCL")

endif()
