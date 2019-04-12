##############################################
#Compute/OpenCL

target_sources(${PROJECT_NAME}
    PRIVATE
        OpenCLBackend.cpp
        OpenCLBackend.h
        OpenCLBuffer.cpp
        OpenCLBuffer.h
        OpenCLDevice.cpp
        OpenCLDevice.h
)