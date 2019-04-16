
##############################################
#Transput

target_sources(${PROJECT_NAME}
    PRIVATE
        Input/Button.h
        Input/Button.cpp
        Input/InputModule.h
        Input/InputModule.cpp
        Input/InputSet.h
        Input/Modules/GLFW/GLFWInputBackend.cpp
        Input/Modules/GLFW/GLFWInputBackend.h
        Input/Modules/Mock/MockInputBackend.cpp
        Input/Modules/Mock/MockInputBackend.h
        Resource/Modules/SPIRV/SPIRVLoader.cpp
        Resource/Modules/SPIRV/SPIRVLoader.h
        Resource/Resource.cpp
        Resource/Resource.h
        Resource/ResourceLoader.h
)
