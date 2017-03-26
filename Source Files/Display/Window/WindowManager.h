#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Metrics.h"
#include <string>
#include <functional>

#include "Display\Layout\Layout.h"

#ifdef WIN32
	#undef CreateWindow
#endif

namespace WindowManager {

	//GLFW needs to be initialized
	void Init(bool*);

	//cleanup all windows
	void Terminate();

	bool ShouldClose();

	void SignelClose();

	void CreateWindow(WindowType, const std::string&,int monitor, uint x, uint y, uint width, uint height, std::function<Layout*(GLFWwindow*, glm::uvec2)>);

	void Draw();

	//callbacks
	void Resize(GLFWwindow *, int, int);
	void Refresh(GLFWwindow*);
	void WindowPos(GLFWwindow *, int, int);
	void Close(GLFWwindow *);
}