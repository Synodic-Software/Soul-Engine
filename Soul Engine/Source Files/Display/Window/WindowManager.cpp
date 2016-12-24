#include "WindowManager.h"
#include <vector>
#include "Utility\Logger.h"
#include "Window.h"

std::vector<Window> windows;
Window* masterWindow = nullptr;

namespace WindowManager {

	namespace detail {
		int monitorCount;
		GLFWmonitor** monitors;
	}

	void Init() {
		detail::monitors = glfwGetMonitors(&detail::monitorCount);

	}

	void Terminate() {
		masterWindow = nullptr;
		for (auto const& win : windows) {
			glfwDestroyWindow(win.windowHandle);
		}
	}

	bool ShouldClose() {
		return glfwWindowShouldClose(masterWindow->windowHandle);
	}

	void SignelClose() {
		glfwSetWindowShouldClose(masterWindow->windowHandle, GLFW_TRUE);
	}

	//the moniter number, and a float from 0-1 of the screen size for each dimension
	void SoulCreateWindow(int monitor, uint x, uint y,uint width, uint height) {
		if (monitor>detail::monitorCount) {
			Logger::Log(ERROR, "The specified moniter '", monitor, "' needs to be less than ", detail::monitorCount);
		}

		GLFWmonitor* monitorIn = detail::monitors[monitor];

		windows.push_back({x,y,width,height,monitorIn});

		if (masterWindow == nullptr) {

			masterWindow = &windows[0];

			//	glfwMakeContextCurrent(Soul::masterWindow);

		//	glfwSetInputMode(windowOut, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		}

	}

	void Draw() {

		for (auto& win : windows) {
			win.Draw();
		}
	}


}