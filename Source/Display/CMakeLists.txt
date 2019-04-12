
##############################################
#Display

target_sources(${PROJECT_NAME}
    PRIVATE
        DisplayModule.cpp
        DisplayModule.h
        Modules/GLFW/GLFWDisplay.cpp
        Modules/GLFW/GLFWDisplay.h
        Modules/GLFW/GLFWWindow.cpp
        Modules/GLFW/GLFWWindow.h
        Modules/Mock/MockDisplay.cpp
        Modules/Mock/MockDisplay.h
        Window.cpp
        Window.h
)
