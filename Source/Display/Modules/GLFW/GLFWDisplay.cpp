#include "GLFWDisplay.h"

#include "Display/WindowParameters.h"
#include "Display/Modules/GLFW/GLFWWindow.h"
#include "Rasterer/Modules/Vulkan/VulkanRasterBackend.h"
#include "Core/Utility/Exception/Exception.h"

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include <cassert>

GLFWDisplay::GLFWDisplay()
{

	//set the error callback
	// TODO: Proper error handling
	glfwSetErrorCallback([](int error, const char* description) {

		throw NotImplemented();

	});

	//Initialize GLFW context for Window handling
	// TODO: proper error handling
	const auto initSuccess = glfwInit();
	assert(initSuccess);

	//Raster API specific checks
	assert(glfwVulkanSupported());

	//TODO: std::span
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

	glfwTerminate();

}

void GLFWDisplay::Draw() {

}

bool GLFWDisplay::Active() {

	return windows_.size() > 0;
	
}

void GLFWDisplay::CreateWindow(const WindowParameters& params, RasterBackend* rasterModule) {

	assert(params.monitor < static_cast<int>(monitors_.size()));

	GLFWmonitor* monitor = monitors_[params.monitor];


	std::unique_ptr<GLFWWindow> window = std::make_unique<GLFWWindow>(params, monitor, rasterModule, windows_.empty());

	const auto context = window->Context();
	windows_[context] = std::move(window);

	//set so the window object that holds the context is visible in callbacks
	glfwSetWindowUserPointer(context, this);

	//all window related callbacks
	glfwSetWindowSizeCallback(context, [](GLFWwindow* window, const int x, const int y)
	{
		const auto display = static_cast<GLFWDisplay*>(glfwGetWindowUserPointer(window));
		display->Resize(x, y);
	});

	glfwSetWindowPosCallback(context, [](GLFWwindow* window, const int x, const int y)
	{
		const auto display = static_cast<GLFWDisplay*>(glfwGetWindowUserPointer(window));
		display->PositionUpdate(x, y);
	});

	glfwSetWindowRefreshCallback(context, [](GLFWwindow* window)
	{
		const auto display = static_cast<GLFWDisplay*>(glfwGetWindowUserPointer(window));
		display->Refresh();
	});

	glfwSetFramebufferSizeCallback(context, [](GLFWwindow* window, const int x, const int y)
	{
		const auto display = static_cast<GLFWDisplay*>(glfwGetWindowUserPointer(window));
		auto& thisWindow = display->GetWindow(window);

		//Only resize if necessary
		if (static_cast<uint>(x) != thisWindow.Parameters().pixelSize.x || static_cast<uint>(y) != thisWindow.Parameters().pixelSize.y) {
			display->FrameBufferResize(x, y);
		}
	});

	glfwSetWindowCloseCallback(context, [](GLFWwindow* window)
	{
		const auto display = static_cast<GLFWDisplay*>(glfwGetWindowUserPointer(window));
		display->Close(display->GetWindow(window));
	});

	//only show the window once all proper callbacks and settings are in place
	glfwShowWindow(context);

}

void GLFWDisplay::RegisterRasterBackend(RasterBackend* rasterBackend)
{

	uint32 glfwExtensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector<const char*> requiredInstanceExtensions;
	requiredInstanceExtensions.reserve(glfwExtensionCount);

	for (uint i = 0; i < glfwExtensionCount; ++i) {
		requiredInstanceExtensions.push_back(glfwExtensions[i]);
	}

	//TODO: pattern match better
	dynamic_cast<VulkanRasterBackend*>(rasterBackend)->AddInstanceExtensions(requiredInstanceExtensions);

}

void GLFWDisplay::Refresh() {

}

void GLFWDisplay::Resize(const int x, const int y) {

}

void GLFWDisplay::FrameBufferResize(const int x, const int y) {

}

void GLFWDisplay::PositionUpdate(const int, const int) {

}

void GLFWDisplay::Close(GLFWWindow& window) {

	if (!window.Master()) {

		windows_.erase(window.Context());

	}
	else
	{

		windows_.clear();

	}
}

GLFWWindow& GLFWDisplay::GetWindow(GLFWwindow* context)
{

	return *windows_[context];

}