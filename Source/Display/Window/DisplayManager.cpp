#include "DisplayManager.h"
#include "Core/Utility/Log/Logger.h"
#include "Desktop/DesktopWindow.h"

DisplayManager::DisplayManager() :
	monitorCount(0),
	runningFlag(true)
{

	//set the error callback
	glfwSetErrorCallback([](int error, const char* description) {
		S_LOG_FATAL("GLFW Error occured, Error ID:", error, " Description:", description);
	});

	//Initialize glfw context for Window handling
	const int didInit = glfwInit();

	if (!didInit) {
		S_LOG_FATAL("GLFW did not initialize");
	}

	monitors = glfwGetMonitors(&monitorCount);

}

DisplayManager::~DisplayManager() {
	masterWindow = nullptr;
	glfwTerminate();
}

void DisplayManager::Draw() {
	for (auto& itr : windows) {
		itr->Draw();
	}
}

bool DisplayManager::ShouldClose() const {
	const auto mainWindow = std::any_cast<GLFWwindow*>(masterWindow->context_);
	if (mainWindow != nullptr) {
		return glfwWindowShouldClose(mainWindow);
	}

	// In the case that there is no window system, this should always return false.
	return false;

}

void DisplayManager::SignalClose() {
	for (auto& win : windows) {
		glfwSetWindowShouldClose(std::any_cast<GLFWwindow*>(win->context_), GLFW_TRUE);
	}

	runningFlag = false;
}

SoulWindow* DisplayManager::CreateWindow(WindowParameters& params) {
	if (params.monitor > monitorCount) {
		S_LOG_ERROR("The specified monitor '", params.monitor, "' needs to be less than ", monitorCount);
		return nullptr;
	}

	void* monitorIn = monitors[params.monitor];

	if (!masterWindow) {
		windows.emplace_back(new DesktopWindow(params, nullptr, nullptr));
		masterWindow = windows.front().get();
	}
	else {
		GLFWwindow* sharedCtx = std::any_cast<GLFWwindow*>(masterWindow->context_);
		windows.emplace_back(new DesktopWindow(params, monitorIn, sharedCtx));
	}

	return windows.back().get();
}

void DisplayManager::Refresh() {

}

void DisplayManager::Resize(int, int) {

}

void DisplayManager::WindowPos(int, int) {

}