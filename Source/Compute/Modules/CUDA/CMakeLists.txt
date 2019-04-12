##############################################
#Compute/CUDA

target_sources(${PROJECT_NAME}
    PRIVATE
        CUDABackend.cpp
        CUDABackend.h
        CUDABuffer.cpp
        CUDABuffer.h
        CUDADevice.cpp
        CUDADevice.h
)