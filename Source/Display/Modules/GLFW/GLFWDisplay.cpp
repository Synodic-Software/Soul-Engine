#include "GLFWDisplay.h"

#include "Display/WindowParameters.h"
#include "Display/Modules/GLFW/GLFWWindow.h"
#include "Rasterer/Modules/Vulkan/VulkanRasterBackend.h"

#include <cassert>


GLFWDisplay::GLFWDisplay()
{

	//set the error callback
	// TODO: Proper error handling
	glfwSetErrorCallback([](int error, const char* description) {
		assert(false);
	});

	//Initialize GLFW context for Window handling
	// TODO: proper error handling
	const auto didInit = glfwInit();
	assert(didInit);

	//Raster API specific checks
	assert(glfwVulkanSupported());

	//TODO: abstract monitors
	int monitorCount;
	GLFWmonitor** tempMonitors = glfwGetMonitors(&monitorCount);
	monitors_.reserve(monitorCount);

	for (auto i = 0; i < monitorCount; ++i)
	{
		monitors_.push_back(tempMonitors[i]);
	}

	//Global GLFW window settings
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); //hide the created windows until they are ready after all callbacks and hints are finished. 
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); //OpenGL is not used

}

GLFWDisplay::~GLFWDisplay()
{

	windows_.clear();
	masterWindow_.reset();

	glfwTerminate();

}

void GLFWDisplay::Draw() {

}

bool GLFWDisplay::Active() {

	if (masterWindow_) {
		return !glfwWindowShouldClose(masterWindow_->Context());
	}

	//If there is no master window...
	return true;
	
}

std::shared_ptr<Window> GLFWDisplay::CreateWindow(WindowParameters& params, std::shared_ptr<RasterBackend> rasterModule) {

	assert(params.monitor < static_cast<int>(monitors_.size()));

	GLFWmonitor* monitor = monitors_[params.monitor];

	std::shared_ptr<GLFWWindow> window = std::make_shared<GLFWWindow>(params, monitor, std::static_pointer_cast<VulkanRasterBackend>(rasterModule)->GetInstance());

	if (!masterWindow_) {
		masterWindow_ = window;
	}

	return window;

}

std::vector<char const*> GLFWDisplay::GetRequiredExtensions()
{

	uint32 glfwExtensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector<const char*> requiredInstanceExtensions;
	requiredInstanceExtensions.reserve(glfwExtensionCount);

	for (uint i = 0; i < glfwExtensionCount; ++i) {
		requiredInstanceExtensions.push_back(glfwExtensions[i]);
	}

	return requiredInstanceExtensions;

}