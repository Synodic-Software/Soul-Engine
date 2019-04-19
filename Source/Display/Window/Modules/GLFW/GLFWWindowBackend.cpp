#include "GLFWWindowBackend.h"

#include "WindowParameters.h"
#include "Display/Window/Modules/GLFW/GLFWWindow.h"
#include "Rasterer/Modules/Vulkan/VulkanRasterBackend.h"
#include "Core/Utility/Exception/Exception.h"

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include <cassert>

GLFWWindowBackend::GLFWWindowBackend()
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

GLFWWindowBackend::~GLFWWindowBackend()
{

	windows_.clear();

	glfwTerminate();

}

void GLFWWindowBackend::Draw() {

	//TODO: selective drawing based on dirty state
	for (auto& window : windows_)
	{

		window.second->Draw();

	}

}

bool GLFWWindowBackend::Active() {

	return !windows_.empty();

}

void GLFWWindowBackend::CreateWindow(const WindowParameters& params, RasterModule* rasterModule)
{

	assert(params.monitor < static_cast<int>(monitors_.size()));

	GLFWmonitor* monitor = monitors_[params.monitor];


	std::unique_ptr<GLFWWindow> window = std::make_unique<GLFWWindow>(params, monitor, static_cast<VulkanRasterBackend*>(rasterModule), windows_.empty());

	const auto context = window->Context();
	windows_[context] = std::move(window);

	//set so the window object that holds the context is visible in callbacks
	glfwSetWindowUserPointer(context, this);

	//all window related callbacks
	glfwSetWindowSizeCallback(context, [](GLFWwindow* window, const int x, const int y)
	{
		const auto display = static_cast<GLFWWindowBackend*>(glfwGetWindowUserPointer(window));
		display->Resize(x, y);
	});

	glfwSetWindowPosCallback(context, [](GLFWwindow* window, const int x, const int y)
	{
		const auto display = static_cast<GLFWWindowBackend*>(glfwGetWindowUserPointer(window));
		display->PositionUpdate(x, y);
	});

	glfwSetWindowRefreshCallback(context, [](GLFWwindow* window)
	{
		const auto display = static_cast<GLFWWindowBackend*>(glfwGetWindowUserPointer(window));
		display->Refresh();
	});

	glfwSetFramebufferSizeCallback(context, [](GLFWwindow* window, const int x, const int y)
	{
		const auto display = static_cast<GLFWWindowBackend*>(glfwGetWindowUserPointer(window));
		auto& thisWindow = display->GetWindow(window);

		//Only resize if necessary
		if (static_cast<uint>(x) != thisWindow.Parameters().pixelSize.x || static_cast<uint>(y) != thisWindow.Parameters().pixelSize.y) {
			display->FrameBufferResize(thisWindow, x, y);
		}
	});

	glfwSetWindowCloseCallback(context, [](GLFWwindow* window)
	{
		const auto display = static_cast<GLFWWindowBackend*>(glfwGetWindowUserPointer(window));
		display->Close(display->GetWindow(window));
	});

	//only show the window once all proper callbacks and settings are in place
	glfwShowWindow(context);

}

void GLFWWindowBackend::RegisterRasterBackend(RasterModule* rasterBackend)
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

void GLFWWindowBackend::Refresh() {

}

void GLFWWindowBackend::Resize(const int x, const int y) {

}

void GLFWWindowBackend::FrameBufferResize(GLFWWindow& window, const int x, const int y) {

	window.FrameBufferResize(x, y);

}

void GLFWWindowBackend::PositionUpdate(const int, const int) {

}

void GLFWWindowBackend::Close(GLFWWindow& window) {

	if (!window.Master()) {

		windows_.erase(window.Context());

	}
	else
	{

		windows_.clear();

	}
}

GLFWWindow& GLFWWindowBackend::GetWindow(GLFWwindow* context)
{

	return *windows_[context];

}