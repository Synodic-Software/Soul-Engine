#include "GLFWWindow.h"

#include "Rasterer/Modules/Vulkan/VulkanRasterBackend.h"

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

GLFWWindow::GLFWWindow(const WindowParameters& params, GLFWmonitor* monitor, RasterBackend* rasterBackend, bool master) :
	Window(params),
	master_(master)
{

	const GLFWvidmode* mode = glfwGetVideoMode(monitor);

	glfwWindowHint(GLFW_RED_BITS, mode->redBits);
	glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
	glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
	glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

	//TODO Cleanup and implement more features

	//defaulted params
	GLFWmonitor* fullscreenMonitor = nullptr;

	//specific window type creation settings, each setting is global, so all should be set in each possibility
	if (windowParams_.type == WindowType::FULLSCREEN) {

		glfwWindowHint(GLFW_RESIZABLE, false);
		glfwWindowHint(GLFW_DECORATED, false);

		fullscreenMonitor = monitor;

	}
	else if (windowParams_.type == WindowType::WINDOWED) {

		glfwWindowHint(GLFW_RESIZABLE, true);
		glfwWindowHint(GLFW_DECORATED, true);

	}
	else if (windowParams_.type == WindowType::BORDERLESS) {

		glfwWindowHint(GLFW_RESIZABLE, false);
		glfwWindowHint(GLFW_DECORATED, false);

	}
	else {

		glfwWindowHint(GLFW_RESIZABLE, false);
		glfwWindowHint(GLFW_DECORATED, true);

	}

	context_ = glfwCreateWindow(windowParams_.pixelSize.x, windowParams_.pixelSize.y, windowParams_.title.c_str(), fullscreenMonitor, nullptr);
	assert(context_);

	//context related settings
	glfwSetInputMode(context_, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

	//GLFW guarantees Vulkan support.
	VkSurfaceKHR castSurface;

	//GLFW is guaranteed to use Vulkan
	auto vulkanRasterBackend = dynamic_cast<VulkanRasterBackend*>(rasterBackend);

	//guaranteed to use GLFW if using Vulkan
	const VkResult error = glfwCreateWindowSurface(
		static_cast<VkInstance>(vulkanRasterBackend->GetInstance()),
		context_,
		nullptr,
		&castSurface
	);

	assert(error == VK_SUCCESS);

	vulkanRasterBackend->RegisterSurface(castSurface);

}

GLFWWindow::~GLFWWindow() {

	glfwDestroyWindow(context_);

}

GLFWwindow* GLFWWindow::Context() const
{

	return context_;

}

bool GLFWWindow::Master() const
{
	
	return master_;

}