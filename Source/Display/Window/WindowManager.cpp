#include "WindowManager.h"

#include "Utility\Logger.h"
#include "Transput\Settings.h"

#include "Raster Engine\RasterManager.h"

#include <memory>


WindowManager::WindowManager() :
	monitorCount(0),
	monitors(nullptr),
	runningFlag(true)
{

	monitors = glfwGetMonitors(&monitorCount);

}

WindowManager::~WindowManager()
{

	masterWindow = nullptr;

}

/*
 *    Determine if we should close.
 *    @return	True if it succeeds, false if it fails.
 */

bool WindowManager::ShouldClose() {
	if (masterWindow != nullptr) {
		return glfwWindowShouldClose(masterWindow->windowHandle);
	}
	else {
		//in the case that there is no window system, this should always return false
		return false;
	}
}

/* Signel close. */
void WindowManager::SignelClose() {

	for (auto& win : windows) {
		glfwSetWindowShouldClose(win->windowHandle, GLFW_TRUE);
	}

	runningFlag = false;

}

/*
 *    the moniter number.
 *    @param	type   	The type.
 *    @param	name   	The name.
 *    @param	monitor	The monitor.
 *    @param	x	   	An uint to process.
 *    @param	y	   	An uint to process.
 *    @param	width  	The width.
 *    @param	height 	The height.
 *    @return	Null if it fails, else the new window.
 */

Window* WindowManager::CreateWindow(WindowType type, const std::string& name, int monitor, uint x, uint y, uint width, uint height) {

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
 *    @param [in,out]	window	If non-null, the window.
 *    @param [in,out]	layout	If non-null, the layout.
 */

void WindowManager::SetWindowLayout(Window* window, Layout* layout) {
	window->layout.reset(layout);
	window->layout->UpdateWindow(windows.back().get()->windowHandle);
	window->layout->UpdatePositioning(glm::uvec2(window->xPos, window->yPos), glm::uvec2(window->width, window->height));
	window->layout->RecreateData();
}



/* Draws this object. */
void WindowManager::Draw() {

	for (auto& itr : windows) {
		itr->Draw();
	}

}

/*
 *    Closes the given handler.
 *    @param [in,out]	handler	If non-null, the handler.
 */

void WindowManager::Close(GLFWwindow* handler) {

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
 *    @param [in,out]	handler	If non-null, the handler.
 *    @param 		 	width  	The width.
 *    @param 		 	height 	The height.
 */

void WindowManager::Resize(GLFWwindow* handler, int width, int height)
{
	RasterManager::Instance().ResizeWindow(handler, width, height);
}

/*
 *    Window position.
 *    @param [in,out]	handler	If non-null, the handler.
 *    @param 		 	x	   	The x coordinate.
 *    @param 		 	y	   	The y coordinate.
 */

void WindowManager::WindowPos(GLFWwindow* handler, int x, int y)
{
}

/*
 *    Refreshes the given handler.
 *    @param [in,out]	handler	If non-null, the handler.
 */

void WindowManager::Refresh(GLFWwindow* handler)
{

	/*Window* window = static_cast<Window*>(glfwGetWindowUserPointer(handler));
	window->Draw();*/
}
