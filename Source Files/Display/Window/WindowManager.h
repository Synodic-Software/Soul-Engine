#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Metrics.h"
#include <string>
#include <functional>
#include "Window.h"
#include "Display\Layout\Layout.h"

#ifdef WIN32
#undef CreateWindow
#endif

namespace WindowManager {

	//GLFW needs to be initialized
	void Initialize(bool*);

	//cleanup all windows
	void Terminate();

	bool ShouldClose();

	void SignelClose();

	Window* CreateWindow(WindowType, const std::string&, int monitor, uint x, uint y, uint width, uint height);

	void SetWindowLayout(Window*, Layout*);

	void Draw();

	//callbacks
	void Resize(GLFWwindow *, int, int);
	void Refresh(GLFWwindow*);
	void WindowPos(GLFWwindow *, int, int);
	void Close(GLFWwindow *);
}