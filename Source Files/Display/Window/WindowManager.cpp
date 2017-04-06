#include "WindowManager.h"

#include "Utility\Logger.h"
#include "Window.h"
#include "Utility\Settings.h"
#include "Multithreading\Scheduler.h"
#include "Raster Engine\RasterBackend.h"

#include "Utility\Includes\GLMIncludes.h"

#include <string>
#include <vector>
#include <mutex>
#include <memory>

static std::list<std::unique_ptr<Window>> windows;
static Window* masterWindow = nullptr;

static int monitorCount;
static GLFWmonitor** monitors;

static bool* runningFlag;

namespace WindowManager {

	void Init(bool* runningFlagIn) {

		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
			RasterBackend::Init();
		});

		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, []() {
			monitors = glfwGetMonitors(&monitorCount);
		});

		
		runningFlag = runningFlagIn;

		Scheduler::Block();

		//windows.emplace_back(new Window(static_cast<WindowType>(-1), "", 0, 0, 1, 1, monitors[0], nullptr));
		//masterWindow = windows.back().get();
	}

	void Terminate() {

		windows.clear();

		RasterBackend::Terminate();

		masterWindow = nullptr;
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

		for (auto& win : windows) {
			Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, [&win]() {
				glfwSetWindowShouldClose(win->windowHandle, GLFW_TRUE);
			});
		}

		*runningFlag = false;

		Scheduler::Block();
	}

	//the moniter number
	void CreateWindow(WindowType type, const std::string& name, int monitor, uint x, uint y, uint width, uint height, std::function<Layout*()> createLayout) {

		if (monitor > monitorCount) {
			S_LOG_ERROR("The specified moniter '", monitor, "' needs to be less than ", monitorCount);
			return;
		}

		GLFWmonitor* monitorIn = monitors[monitor];

		if (!masterWindow) {
			windows.emplace_back(new Window(type, name, x, y, width, height, monitorIn, nullptr));
			masterWindow = windows.front().get();
		}
		else {
			GLFWwindow* sharedCtx = masterWindow->windowHandle;
			windows.emplace_back(new Window(type, name, x, y, width, height, monitorIn, sharedCtx));
		}

		windows.back()->layout.reset(createLayout());
		windows.back()->layout->UpdateWindow(windows.back().get()->windowHandle);
		windows.back()->layout->UpdatePositioning( glm::uvec2(x,y), glm::uvec2(width, height));
		windows.back()->layout->RecreateData();

	}



	void Draw() {

		for (auto& itr : windows) {
			Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, [&itr]() {
				itr->Draw();
			});

		}

		Scheduler::Block();
	}

	void Close(GLFWwindow* handler) {

		Window* window = static_cast<Window*>(glfwGetWindowUserPointer(handler));

		if (masterWindow == window) {
			SignelClose();
			masterWindow = nullptr;
		}


		// Find the matching unique_ptr
		auto b = windows.begin();
		auto e = windows.end();
		while (b != e)
		{
			if (window == b->get())
				b = windows.erase(b);
			else
				++b;
		}

	}


	void Resize(GLFWwindow* handler, int width, int height)
	{
		RasterBackend::ResizeWindow(handler, width, height);
	}
	 
	void WindowPos(GLFWwindow* handler, int x, int y)
	{
	}

	void Refresh(GLFWwindow* handler)
	{

		Window* window = static_cast<Window*>(glfwGetWindowUserPointer(handler));
		window->Draw();
	}
}