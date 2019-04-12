
##############################################
#Transput

target_sources(${PROJECT_NAME}
    PRIVATE
        Input/Console/CLI/CLIConsoleManager.cpp
        Input/Console/CLI/CLIConsoleManager.h
        Input/Console/ConsoleManager.cpp
        Input/Console/ConsoleManager.h
        Input/Console/Parser/CommandParser.cpp
        Input/Console/Parser/CommandParser.h
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
