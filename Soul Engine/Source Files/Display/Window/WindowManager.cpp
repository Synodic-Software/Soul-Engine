#include "WindowManager.h"

#include <vector>
#include "Utility\Logger.h"
#include "Window.h"
#include "Utility\Settings.h"
#include "Multithreading\Scheduler.h"
#include "Raster Engine\RasterBackend.h"

namespace {
	std::vector<Window> windows;
	Window* masterWindow = nullptr;

	int monitorCount;
	GLFWmonitor** monitors;
}

namespace WindowManager {

	void Init() {

		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&]() {
			monitors = glfwGetMonitors(&monitorCount);
		});

		Scheduler::Block();

		uint xSize = Settings::Get("MainWindow.Width", 1024);
		uint ySize = Settings::Get("MainWindow.Height", 720);
		uint xPos = Settings::Get("MainWindow.X_Position", 0);
		uint yPos = Settings::Get("MainWindow.Y_Position", 0);

		windows.emplace_back(std::string("Main"), xPos, yPos, xSize, ySize, monitors[0], nullptr);

		masterWindow = &windows[0];
	}

	void Terminate() {
		masterWindow = nullptr;
		for (auto const& win : windows) {
			Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&]() {
				glfwDestroyWindow(win.windowHandle);
			});
		}

		Scheduler::Block();
	}

	bool ShouldClose() {
		if (masterWindow != nullptr) {
			return glfwWindowShouldClose(masterWindow->windowHandle);
		}
		else {
			S_LOG_WARNING("No window has been created");
			return false;
		}
	}

	void SignelClose() {
		if (masterWindow != nullptr) {
			glfwSetWindowShouldClose(masterWindow->windowHandle, GLFW_TRUE);
		}
		else {
			S_LOG_WARNING("No window has been created");
		}
	}

	//the moniter number
	void SCreateWindow(std::string& name, int monitor, uint x, uint y, uint width, uint height) {

		if (monitor > monitorCount) {
			S_LOG_ERROR("The specified moniter '", monitor, "' needs to be less than ", monitorCount);
		}

		GLFWmonitor* monitorIn = monitors[monitor];

		windows.emplace_back(name, x, y, width, height, monitorIn, masterWindow->windowHandle);

	}

	void Draw() {

		for (Window& win : windows) {
			Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, [&]() {
				win.Draw();
			});
		}

		Scheduler::Block();


		RasterBackend::Draw();


	}


}