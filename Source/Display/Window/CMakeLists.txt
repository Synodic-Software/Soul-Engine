
##############################################
#Window

target_sources(${PROJECT_NAME}
    PRIVATE
        WindowModule.cpp
        WindowModule.h
        Modules/GLFW/GLFWWindowBackend.cpp
        Modules/GLFW/GLFWWindowBackend.h
        Modules/GLFW/GLFWWindow.cpp
        Modules/GLFW/GLFWWindow.h
        Modules/Mock/MockWindowBackend.cpp
        Modules/Mock/MockWindowBackend.h
        Modules/Mock/MockWindow.cpp
        Modules/Mock/MockWindow.h
        Window.cpp
        Window.h
)
