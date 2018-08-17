#include "DesktopWindowManager.h"

#include "Core/Utility/Log/Logger.h"
#include "Display/Window/Desktop/DesktopWindow.h"

#include "GLFW/glfw3.h"


DesktopWindowManager::DesktopWindowManager(EntityManager& entityManager, DesktopInputManager& inputManager, RasterManager& rasterManager) :
	entityManager_(&entityManager),
	inputManager_(&inputManager),
	rasterManager_(&rasterManager)
{

	//set the error callback
	// TODO: Proper error handling
	glfwSetErrorCallback([](int error, const char* description) {
		S_LOG_FATAL("GLFW Error occured, Error ID:", error, " Description:", description);
		assert(false);
	});

	//Initialize glfw context for Window handling
	// TODO: proper error handling
	const auto didInit = glfwInit();

	assert(didInit);

	//TODO: abstract monitors
	monitors_ = glfwGetMonitors(&monitorCount_);

	//global GLFW settings
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); //hide the created windows until they are ready after all callbacks and hints are finished. 
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); //OpenGL is not used

}

void DesktopWindowManager::Terminate() {

	entityManager_->RemoveComponent<DesktopWindow>();
	windows_.clear();
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

	for (auto& winEntity : windows_) {

		auto& win = entityManager_->GetComponent<DesktopWindow>(winEntity);

		glfwSetWindowShouldClose(std::any_cast<GLFWwindow*>(win.GetContext()), GLFW_TRUE);

	}

	runningFlag_ = false;

}

Window& DesktopWindowManager::CreateWindow(WindowParameters& params) {

	assert(params.monitor < monitorCount_);

	GLFWmonitor* monitor = monitors_[params.monitor];

	const Entity newEntity = entityManager_->CreateEntity();
	windows_.push_back(newEntity);

	entityManager_->AttachComponent<DesktopWindow>(newEntity, entityManager_, params, monitor, *inputManager_, *rasterManager_);

	if (masterWindow_.IsNull()) {
		masterWindow_ = newEntity;
	}

	return entityManager_->GetComponent<DesktopWindow>(newEntity);

}

void DesktopWindowManager::Draw() {

	for (auto& windowEntity : windows_) {

		auto& window = entityManager_->GetComponent<DesktopWindow>(windowEntity);
		window.Draw();

	}

}