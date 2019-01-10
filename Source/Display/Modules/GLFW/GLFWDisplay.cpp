#include "GLFWDisplay.h"

#include "Core/Utility/Exception/Exception.h"
#include "Display/WindowParameters.h"
#include "Display/Modules/GLFW/GLFWWindow.h"

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

	//TODO: abstract monitors
	int monitorCount;
	GLFWmonitor** tempMonitors = glfwGetMonitors(&monitorCount);
	monitors_.reserve(monitorCount);

	for (auto i = 0; i < monitorCount; ++i)
	{
		monitors_.push_back(tempMonitors[i]);
	}

	//Initial GLFW settings
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); //hide the created windows until they are ready after all callbacks and hints are finished. 
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); //OpenGL is not used

}

GLFWDisplay::~GLFWDisplay()
{

	glfwTerminate();

}

void GLFWDisplay::Draw() {

	throw NotImplemented();

}

bool GLFWDisplay::ShouldClose() {

	if (masterWindow_) {
		return glfwWindowShouldClose(masterWindow_->Context());
	}

	return false;

}

std::shared_ptr<Window> GLFWDisplay::CreateWindow(WindowParameters& params) {

	assert(params.monitor < monitors_.size());

	GLFWmonitor* monitor = monitors_[params.monitor];

	std::shared_ptr<GLFWWindow> window = std::make_shared<GLFWWindow>(params, monitor);

	if (!window) {
		masterWindow_ = window;
	}

	return window;

}