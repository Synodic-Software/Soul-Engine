#include "DesktopWindowManager.h"

#include "Core/Utility/Log/Logger.h"
#include "Display/Window/Desktop/DesktopWindow.h"

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

DesktopWindowManager::DesktopWindowManager(EntityManager& entityManager,DesktopInputManager& inputManager):
	WindowManager(entityManager),
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
	/*for (auto& itr : windows_) {
		itr->Draw();
	}*/
}

bool DesktopWindowManager::ShouldClose() const {

	auto& win = entityManager_->GetComponent<DesktopWindow>(masterWindow_);
	const auto mainWindow = std::any_cast<GLFWwindow*>(win.GetContext());
	if (mainWindow != nullptr) {
		return glfwWindowShouldClose(mainWindow);
	}

	// In the case that there is no window system, this should always return false.
	return false;

}

void DesktopWindowManager::SignalClose() {
	for (auto& windowEntity : windows_) {
		DesktopWindow& win= entityManager_->GetComponent<DesktopWindow>(windowEntity);
		glfwSetWindowShouldClose(std::any_cast<GLFWwindow*>(win.GetContext()), GLFW_TRUE);
	}

	runningFlag_ = false;
}

Window& DesktopWindowManager::CreateWindow(WindowParameters& params) {
	if (params.monitor > monitorCount_) {
		S_LOG_ERROR("The specified monitor '", params.monitor, "' needs to be less than ", monitorCount_);
	}

	GLFWmonitor* monitor = monitors_[params.monitor];


	const Entity windowEntity = entityManager_->CreateEntity();

	if (masterWindow_.IsNull()) {

		entityManager_->AttachComponent<DesktopWindow>(windowEntity, params, monitor, nullptr, *inputManager_);
		masterWindow_ = windowEntity;

	}
	else {
		auto& win = entityManager_->GetComponent<DesktopWindow>(masterWindow_);
		const auto sharedCtx = std::any_cast<GLFWwindow*>(win.GetContext());

		entityManager_->AttachComponent<DesktopWindow>(windowEntity, params, monitor, sharedCtx, *inputManager_);
	}

	windows_.push_back(windowEntity);

	return entityManager_->GetComponent<DesktopWindow>(windowEntity);
}

void DesktopWindowManager::Refresh() {

}

void DesktopWindowManager::Resize(int, int) {

}

void DesktopWindowManager::WindowPos(int, int) {

}