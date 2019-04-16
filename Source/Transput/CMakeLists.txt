
##############################################
#Transput

target_sources(${PROJECT_NAME}
    PRIVATE
        Input/Desktop/DesktopInputManager.cpp
        Input/Desktop/DesktopInputManager.h
        Input/InputManager.cpp
        Input/InputManager.h
        Input/InputSet.cpp
        Input/InputSet.h
        Input/Key.cpp
        Input/Key.h
        Resource/Modules/SPIRV/SPIRVLoader.cpp
        Resource/Modules/SPIRV/SPIRVLoader.h
        Resource/Resource.cpp
        Resource/Resource.h
        Resource/ResourceLoader.h
        Serialization/Serializable.cpp
        Serialization/Serializable.h
)
