#include "DesktopWindowManager.h"

#include "Core/Utility/Log/Logger.h"
#include "Display/Window/Desktop/DesktopWindow.h"

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

DesktopWindowManager::DesktopWindowManager(DesktopInputManager& inputManager):
	inputManager_(&inputManager)
{

	//set the error callback
	// TODO: Proper error handling
	glfwSetErrorCallback([](int error, const char* description) {
		S_LOG_FATAL("GLFW Error occured, Error ID:", error, " Description:", description);
	});

	//Initialize glfw context for Window handling
	// TODO: proper error handling
	if (!glfwInit()) {
		S_LOG_FATAL("GLFW did not initialize");
	}

	//TODO: abstract monitors
	monitors_ = glfwGetMonitors(&monitorCount_);

	//global GLFW settings
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); //hide the created windows until they are ready after all callbacks and hints are finished. 
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); //OpenGL is not used, so init a GL context

}

DesktopWindowManager::~DesktopWindowManager() {
	glfwTerminate();
}

void DesktopWindowManager::Draw() {
	for (auto& itr : windows_) {
		itr->Draw();
	}
}

bool DesktopWindowManager::ShouldClose() const {
	const auto mainWindow = std::any_cast<GLFWwindow*>(masterWindow_->GetContext());
	if (mainWindow != nullptr) {
		return glfwWindowShouldClose(mainWindow);
	}

	// In the case that there is no window system, this should always return false.
	return false;

}

void DesktopWindowManager::SignalClose() {
	for (auto& win : windows_) {
		glfwSetWindowShouldClose(std::any_cast<GLFWwindow*>(win->GetContext()), GLFW_TRUE);
	}

	runningFlag_ = false;
}

Window* DesktopWindowManager::CreateWindow(WindowParameters& params) {
	if (params.monitor > monitorCount_) {
		S_LOG_ERROR("The specified monitor '", params.monitor, "' needs to be less than ", monitorCount_);
	}

	GLFWmonitor* monitor = monitors_[params.monitor];

	if (!masterWindow_) {
		windows_.push_back(std::make_unique<DesktopWindow>(params, monitor, nullptr, *inputManager_));
		masterWindow_ = windows_.front().get();
	}
	else {
		const auto sharedCtx = std::any_cast<GLFWwindow*>(masterWindow_->GetContext());
		windows_.push_back(std::make_unique<DesktopWindow>(params, monitor, sharedCtx, *inputManager_));
	}

	return windows_.back().get();
}

void DesktopWindowManager::Refresh() {

}

void DesktopWindowManager::Resize(int, int) {

}

void DesktopWindowManager::WindowPos(int, int) {

}