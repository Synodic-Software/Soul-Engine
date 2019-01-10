#include "GLFWWindow.h"

GLFWWindow::GLFWWindow(WindowParameters& params, GLFWmonitor* monitor) :
	Window(params)
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

	//set so the window object that holds the context is visible in callbacks
	glfwSetWindowUserPointer(context_, this);

	//all window related callbacks
	glfwSetWindowSizeCallback(context_, [](GLFWwindow* w, int x, int y)
	{
		const auto thisWindow = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
		thisWindow->Resize(x, y);
	});

	glfwSetWindowPosCallback(context_, [](GLFWwindow* w, int x, int y)
	{
		const auto thisWindow = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
		thisWindow->PositionUpdate(x, y);
	});

	glfwSetWindowRefreshCallback(context_, [](GLFWwindow* w)
	{
		const auto thisWindow = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
		thisWindow->Refresh();
	});

	glfwSetFramebufferSizeCallback(context_, [](GLFWwindow* w, int x, int y)
	{
		const auto thisWindow = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));

		//Only resize if necessary
		if (x != thisWindow->windowParams_.pixelSize.x || y != thisWindow->windowParams_.pixelSize.y) {
			thisWindow->FrameBufferResize(x, y);
		}
	});

	glfwSetWindowCloseCallback(context_, [](GLFWwindow* w)
	{
		const auto thisWindow = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
		thisWindow->Close();
	});

	//only show the window once all proper callbacks and settings are in place
	glfwShowWindow(context_);

}

GLFWWindow::~GLFWWindow() {

	glfwDestroyWindow(context_);

}

void GLFWWindow::Refresh() {

}

void GLFWWindow::Close() {

}

void GLFWWindow::Resize(int x, int y) {

}

void GLFWWindow::FrameBufferResize(int x, int y) {

}

void GLFWWindow::PositionUpdate(int, int) {

}

GLFWwindow* GLFWWindow::Context()
{

	return context_;

}