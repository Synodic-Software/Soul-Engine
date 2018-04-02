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

DesktopManager::~DesktopManager()
{
	masterWindow = nullptr;
}

AbstractWindow* DesktopManager::CreateWindow(WindowType type, const std::string& name, int monitor, uint x, uint y, uint width, uint height)
{
	if (monitor > monitorCount) {
		S_LOG_ERROR("The specified moniter '", monitor, "' needs to be less than ", monitorCount);
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

void DesktopManager::Close(void*)
{
}

void DesktopManager::Draw()
{
}

void DesktopManager::Resize(void*, int, int)
{
}

void DesktopManager::Refresh(void*)
{
}

void DesktopManager::WindowPos(void*, int, int)
{
}

