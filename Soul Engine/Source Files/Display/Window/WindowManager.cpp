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
		SoulCreateWindow("Main", 0, 0, 0, 1024, 720);
	}

	void Terminate() {
		masterWindow = nullptr;
		for (auto const& win : windows) {
			glfwDestroyWindow(win.windowHandle);
		}
	}

	bool ShouldClose() {
		if (masterWindow!=nullptr) {
			return glfwWindowShouldClose(masterWindow->windowHandle);
		}
		else {
			LOG_WARNING("No window has been created");
			return false;
		}
	}

	void SignelClose() {
		if (masterWindow != nullptr) {
			glfwSetWindowShouldClose(masterWindow->windowHandle, GLFW_TRUE);
		}
		else {
			LOG_WARNING("No window has been created");
		}
	}

	//the moniter number
	void SoulCreateWindow(const std::string& name,int monitor, uint x, uint y,uint width, uint height) {
		if (monitor>detail::monitorCount) {
			LOG(ERROR, "The specified moniter '", monitor, "' needs to be less than ", detail::monitorCount);
		}

		GLFWmonitor* monitorIn = detail::monitors[monitor];

		windows.push_back({ name,x,y,width,height,monitorIn});

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

		for (auto& win : windows) {
			glfwSwapBuffers(win.windowHandle);
		}

	}


}