#include "DesktopWindowManager.h"

#include "Core/Utility/Log/Logger.h"
#include "Display/Window/Desktop/DesktopWindow.h"

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

DesktopWindowManager::DesktopWindowManager(EntityManager& entityManager, DesktopInputManager& inputManager) :
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
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); //OpenGL is not used

}

DesktopWindowManager::~DesktopWindowManager() {

	glfwTerminate();

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

		auto& win = entityManager_->GetComponent<DesktopWindow>(windowEntity);
		glfwSetWindowShouldClose(std::any_cast<GLFWwindow*>(win.GetContext()), GLFW_TRUE);

	}

	runningFlag_ = false;

}

Window& DesktopWindowManager::CreateWindow(WindowParameters& params) {

	assert(params.monitor < monitorCount_);

	GLFWmonitor* monitor = monitors_[params.monitor];
	const Entity windowEntity = entityManager_->CreateEntity();

	if (masterWindow_.IsNull()) {
		masterWindow_ = windowEntity;
	}

	entityManager_->AttachComponent<DesktopWindow>(windowEntity, params, monitor, *inputManager_, *entityManager_);
	windows_.push_back(windowEntity);

	return entityManager_->GetComponent<DesktopWindow>(windowEntity);

}