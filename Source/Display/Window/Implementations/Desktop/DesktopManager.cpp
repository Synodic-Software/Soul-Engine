#include "DesktopManager.h"
#include "DesktopWindow.h"

#include "Utility\Logger.h"
#include "Transput\Settings.h"

#include "Raster Engine\RasterManager.h"

#include <memory>

DesktopManager::DesktopManager() : AbstractManager()
{
	monitors = glfwGetMonitors(&monitorCount);
}

AbstractWindow* DesktopManager::CreateWindow(WindowType type, const std::string& name, int monitor, uint x, uint y, uint width, uint height)
{
	if (monitor > monitorCount) {
		S_LOG_ERROR("The specified monitor '", monitor, "' needs to be less than ", monitorCount);
		return nullptr;
	}

	void* monitorIn = monitors[monitor];

	if (!masterWindow) {
		windows.emplace_back(new DesktopWindow(type, name, x, y, width, height, nullptr, nullptr));
		masterWindow = windows.front().get();
	} else {
		GLFWwindow* sharedCtx = static_cast<GLFWwindow*>(masterWindow->windowHandle);
		windows.emplace_back(new DesktopWindow(type, name, x, y, width, height, monitorIn, sharedCtx));
	}

	return windows.back().get();
}

void DesktopManager::SetWindowLayout(AbstractWindow* window, Layout* layout)
{
	window->layout.reset(layout);
	window->layout->UpdateWindow(static_cast<GLFWwindow*>(windows.back().get()->windowHandle));
	window->layout->UpdatePositioning(glm::uvec2(window->xPos, window->yPos), glm::uvec2(window->width, window->height));
	window->layout->RecreateData();
}

bool DesktopManager::ShouldClose()
{
	return false;
}

void DesktopManager::SignalClose()
{
}


/*
*    Closes the given handler.
*    @param [in,out]	handler	If non-null, the handler.
*/
void DesktopManager::Close(void* handler)
{
	DesktopWindow* window = static_cast<DesktopWindow*>(glfwGetWindowUserPointer(static_cast<GLFWwindow*>(handler)));

	if (masterWindow == window) {
		SignalClose();
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



/* Draws this object. */
void DesktopManager::Draw()
{
	for (auto& itr : windows) {
		itr->Draw();
	}
}



/*
*    Resizes.
*    @param [in,out]	handler	If non-null, the handler.
*    @param 		 	width  	The width.
*    @param 		 	height 	The height.
*/
void DesktopManager::Resize(void*, int, int)
{


}




/*
*    Refreshes the given handler.
*    @param [in,out]	handler	If non-null, the handler.
*/
void DesktopManager::Refresh(void* handler)
{
	DesktopWindow* window = static_cast<DesktopWindow*>(glfwGetWindowUserPointer(static_cast<GLFWwindow*>(handler)));
	window->Draw();
}


/*
*    Window position.
*    @param [in,out]	handler	If non-null, the handler.
*    @param 		 	x	   	The x coordinate.
*    @param 		 	y	   	The y coordinate.
*/
void DesktopManager::WindowPos(void*, int, int)
{


}

