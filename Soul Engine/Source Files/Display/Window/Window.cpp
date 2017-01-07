#include "Window.h"
#include "Utility\Logger.h"
#include "Input\Input.h"
#include "Raster Engine\RasterBackend.h"
#include "Multithreading\Scheduler.h"

Window::Window(WindowType inWin, const std::string& inTitle, uint x, uint y, uint iwidth, uint iheight, GLFWmonitor* monitorIn, GLFWwindow* sharedContextin)
{
	windowType = inWin;
	xPos = x;
	yPos = y;
	width = iwidth;
	height = iheight;
	title = inTitle;

	GLFWwindow* sharedContext;
	if (sharedContextin)
	{
		sharedContext = sharedContextin;
	}
	else
	{
		sharedContext = nullptr;
	}

	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this, monitorIn, sharedContext]() {

		RasterBackend::SetWindowHints();
		glfwWindowHint(GLFW_SAMPLES, GLFW_DONT_CARE);
		glfwWindowHint(GLFW_VISIBLE, true);

		const GLFWvidmode* mode = glfwGetVideoMode(monitorIn);

		if (windowType == FULLSCREEN) {

			glfwWindowHint(GLFW_RESIZABLE, false);
			glfwWindowHint(GLFW_DECORATED, false);

			glfwWindowHint(GLFW_RED_BITS, mode->redBits);
			glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
			glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
			glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
			windowHandle = glfwCreateWindow(width, height, title.c_str(), monitorIn, sharedContext);

		}
		else if (windowType == WINDOWED) {

			glfwWindowHint(GLFW_RESIZABLE, true);

			glfwWindowHint(GLFW_RED_BITS, mode->redBits);
			glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
			glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
			glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
			windowHandle = glfwCreateWindow(width, height, title.c_str(), nullptr, sharedContext);

		}
		else if (windowType == BORDERLESS) {

			glfwWindowHint(GLFW_RESIZABLE, false);
			glfwWindowHint(GLFW_DECORATED, false);

			glfwWindowHint(GLFW_RED_BITS, mode->redBits);
			glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
			glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
			glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
			windowHandle = glfwCreateWindow(width, height, title.c_str(), nullptr, sharedContext);

		}
	});

	Scheduler::Block();

	if (windowHandle == nullptr)
	{
		S_LOG_FATAL("Could not Create GLFW Window");
	}

	RasterBackend::BuildWindow(windowHandle);

	//the backend is the new user

	Window* thisWindow = this;

	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this, thisWindow]() {
		glfwSetWindowUserPointer(windowHandle, thisWindow);
		glfwSetWindowPos(windowHandle, xPos, yPos);

		//all window related callbacks
		glfwSetWindowSizeCallback(windowHandle, [](GLFWwindow* w, int x, int y)
		{
			static_cast<Window*>(glfwGetWindowUserPointer(w))->Resize(w, x, y);
		});

		glfwSetWindowPosCallback(windowHandle, [](GLFWwindow* w, int x, int y)
		{
			static_cast<Window*>(glfwGetWindowUserPointer(w))->WindowPos(w, x, y);
		});

		glfwSetWindowRefreshCallback(windowHandle, [](GLFWwindow* w)
		{
			static_cast<Window*>(glfwGetWindowUserPointer(w))->Refresh(w);
		});

		glfwSetKeyCallback(windowHandle, Input::KeyCallback);
		glfwSetScrollCallback(windowHandle, Input::ScrollCallback);
		glfwSetCursorPosCallback(windowHandle, Input::MouseCallback);
	});

	Scheduler::Block();

}


Window::~Window()
{
	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this]() {
		if (windowHandle) {
			glfwDestroyWindow(windowHandle);
		}
	});

	Scheduler::Block();
}

void Window::Resize(GLFWwindow* window, int width, int height)
{
	RasterBackend::ResizeWindow(window, width, height);
}

void Window::WindowPos(GLFWwindow* window, int x, int y)
{
}

void Window::Refresh(GLFWwindow* window)
{
	Draw();
}

void Window::Draw() 
{
	RasterBackend::PreRaster(windowHandle);
	RasterBackend::Draw(windowHandle);
	RasterBackend::PostRaster(windowHandle);
}