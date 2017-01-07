#include "WindowManager.h"

#include <vector>
#include "Utility\Logger.h"
#include "Window.h"
#include "Utility\Settings.h"
#include "Multithreading\Scheduler.h"
#include "Raster Engine\RasterBackend.h"

#include <string>

namespace {
	std::vector<Window> windows;
	Window* masterWindow = nullptr;

	int monitorCount;
	GLFWmonitor** monitors;
}

namespace WindowManager {

	void Init() {

		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, []() {
			monitors = glfwGetMonitors(&monitorCount);
		});

		Scheduler::Block();

		uint xSize = Settings::Get("MainWindow.Width", 1024);
		uint ySize = Settings::Get("MainWindow.Height", 720);
		uint xPos = Settings::Get("MainWindow.X_Position", 0);
		uint yPos = Settings::Get("MainWindow.Y_Position", 0);
		WindowType type = static_cast<WindowType>(Settings::Get("MainWindow.Type", static_cast<int>(WINDOWED)));

		windows.emplace_back(type,"Main", xPos, yPos, xSize, ySize, monitors[0], nullptr);

		masterWindow = &windows[0];
	}

	void Terminate() {
		windows.clear();
	}

	bool ShouldClose() {
		if (masterWindow != nullptr) {
			return glfwWindowShouldClose(masterWindow->windowHandle);
		}
		else {
			//in the case that there is no window system, this should always return false
			return false;
		}
	}

	void SignelClose() {
		if (masterWindow != nullptr) {
			glfwSetWindowShouldClose(masterWindow->windowHandle, GLFW_TRUE);
		}
	}

	//the moniter number
	void SCreateWindow(WindowType type,std::string& name, int monitor, uint x, uint y, uint width, uint height) {

		if (monitor > monitorCount) {
			S_LOG_ERROR("The specified moniter '", monitor, "' needs to be less than ", monitorCount);
		}

		GLFWmonitor* monitorIn = monitors[monitor];

		windows.emplace_back(type,name, x, y, width, height, monitorIn, masterWindow->windowHandle);

	}

	void Draw() {

		for (Window& win : windows) {
			Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, [&win]() {
				win.Draw();
			});
		}

		Scheduler::Block();
	}


}