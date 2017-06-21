#include "WindowManager.h"

#include "Utility\Logger.h"
#include "Utility\Settings.h"
#include "Multithreading\Scheduler.h"
#include "Raster Engine\RasterBackend.h"

#include "Utility\Includes\GLMIncludes.h"

#include <string>
#include <vector>
#include <mutex>
#include <memory>

/* The windows */
/* The windows */
static std::list<std::unique_ptr<Window>> windows;
/* The master window */
/* The master window */
static Window* masterWindow = nullptr;

/* Number of monitors */
/* Number of monitors */
static int monitorCount;
/* The monitors */
/* The monitors */
static GLFWmonitor** monitors;

/* True to running flag */
/* True to running flag */
static bool* runningFlag;

namespace WindowManager {

	/*
	 *    Initializes this object.
	 *
	 *    @param [in,out]	runningFlagIn	If non-null, true to running flag in.
	 */

	void Initialize(bool* runningFlagIn) {

		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
			RasterBackend::Initialize();
		});

		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, []() {
			monitors = glfwGetMonitors(&monitorCount);
		});

		
		runningFlag = runningFlagIn;

		Scheduler::Block();

		//windows.emplace_back(new Window(static_cast<WindowType>(-1), "", 0, 0, 1, 1, monitors[0], nullptr));
		//masterWindow = windows.back().get();
	}

	/* Terminates this object. */
	/* Terminates this object. */
	void Terminate() {

		windows.clear();

		RasterBackend::Terminate();

		masterWindow = nullptr;
	}

	/*
	 *    Determine if we should close.
	 *
	 *    @return	True if it succeeds, false if it fails.
	 */

	bool ShouldClose() {
		if (masterWindow != nullptr) {
			return glfwWindowShouldClose(masterWindow->windowHandle);
		}
		else {
			//in the case that there is no window system, this should always return false
			return false;
		}
	}

	/* Signel close. */
	/* Signel close. */
	void SignelClose() {

		for (auto& win : windows) {
			Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, [&win]() {
				glfwSetWindowShouldClose(win->windowHandle, GLFW_TRUE);
			});
		}

		*runningFlag = false;

		Scheduler::Block();
	}

	/*
	 *    the moniter number.
	 *
	 *    @param	type   	The type.
	 *    @param	name   	The name.
	 *    @param	monitor	The monitor.
	 *    @param	x	   	An uint to process.
	 *    @param	y	   	An uint to process.
	 *    @param	width  	The width.
	 *    @param	height 	The height.
	 *
	 *    @return	Null if it fails, else the new window.
	 */

	Window* CreateWindow(WindowType type, const std::string& name, int monitor, uint x, uint y, uint width, uint height) {

		if (monitor > monitorCount) {
			S_LOG_ERROR("The specified moniter '", monitor, "' needs to be less than ", monitorCount);
			return nullptr;
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

		return windows.back().get();

	}

	/*
	 *    Sets window layout.
	 *
	 *    @param [in,out]	window	If non-null, the window.
	 *    @param [in,out]	layout	If non-null, the layout.
	 */

	void SetWindowLayout(Window* window, Layout* layout) {
		window->layout.reset(layout);
		window->layout->UpdateWindow(windows.back().get()->windowHandle);
		window->layout->UpdatePositioning(glm::uvec2(window->xPos, window->yPos), glm::uvec2(window->width, window->height));
		window->layout->RecreateData();
	}



	/* Draws this object. */
	/* Draws this object. */
	void Draw() {

		for (auto& itr : windows) {
			Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, [&itr]() {
				itr->Draw();
			});

		}

		Scheduler::Block();
	}

	/*
	 *    Closes the given handler.
	 *
	 *    @param [in,out]	handler	If non-null, the handler.
	 */

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

	/*
	 *    Resizes.
	 *
	 *    @param [in,out]	handler	If non-null, the handler.
	 *    @param 		 	width  	The width.
	 *    @param 		 	height 	The height.
	 */

	void Resize(GLFWwindow* handler, int width, int height)
	{
		RasterBackend::ResizeWindow(handler, width, height);
	}

	/*
	 *    Window position.
	 *
	 *    @param [in,out]	handler	If non-null, the handler.
	 *    @param 		 	x	   	The x coordinate.
	 *    @param 		 	y	   	The y coordinate.
	 */

	void WindowPos(GLFWwindow* handler, int x, int y)
	{
	}

	/*
	 *    Refreshes the given handler.
	 *
	 *    @param [in,out]	handler	If non-null, the handler.
	 */

	void Refresh(GLFWwindow* handler)
	{

		/*Window* window = static_cast<Window*>(glfwGetWindowUserPointer(handler));
		window->Draw();*/
	}
}