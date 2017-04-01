#include "Window.h"
#include "Utility\Logger.h"
#include "Input\Input.h"
#include "Raster Engine\RasterBackend.h"
#include "Multithreading\Scheduler.h"
#include "WindowManager.h"

Window::Window(WindowType inWin, const std::string& inTitle, uint x, uint y, uint iwidth, uint iheight, GLFWmonitor* monitorIn, GLFWwindow* sharedContext)
{
	windowType = inWin;
	xPos = x;
	yPos = y;
	width = iwidth;
	height = iheight;
	title = inTitle;

	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this, sharedContext, monitorIn ]() {

		RasterBackend::SetWindowHints();
		glfwWindowHint(GLFW_SAMPLES, GLFW_DONT_CARE);
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

		const GLFWvidmode* mode = glfwGetVideoMode(monitorIn);

		glfwWindowHint(GLFW_RED_BITS, mode->redBits);
		glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
		glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
		glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

		if (windowType == FULLSCREEN) {

			glfwWindowHint(GLFW_RESIZABLE, false);
			glfwWindowHint(GLFW_DECORATED, false);

			windowHandle = glfwCreateWindow(width, height, title.c_str(), monitorIn, sharedContext);

		}
		else if (windowType == WINDOWED) {

			glfwWindowHint(GLFW_RESIZABLE, true);

			windowHandle = glfwCreateWindow(width, height, title.c_str(), nullptr, sharedContext);

		}
		else if (windowType == BORDERLESS) {

			glfwWindowHint(GLFW_RESIZABLE, false);
			glfwWindowHint(GLFW_DECORATED, false);

			windowHandle = glfwCreateWindow(width, height, title.c_str(), nullptr, sharedContext);

		}
		else {
			glfwWindowHint(GLFW_RESIZABLE, false);
			glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

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

		//all window related callbacks
		glfwSetWindowSizeCallback(windowHandle, [](GLFWwindow* w, int x, int y)
		{
			WindowManager::Resize(w, x, y);
		});

		glfwSetWindowPosCallback(windowHandle, [](GLFWwindow* w, int x, int y)
		{
			WindowManager::WindowPos(w, x, y);
		});

		glfwSetWindowRefreshCallback(windowHandle, [](GLFWwindow* w)
		{
			WindowManager::Refresh(w);
		});

		glfwSetWindowCloseCallback(windowHandle, [](GLFWwindow* w)
		{
			WindowManager::Close(w);
		});

		glfwSetKeyCallback(windowHandle, Input::KeyCallback);
		glfwSetScrollCallback(windowHandle, Input::ScrollCallback);
		glfwSetCursorPosCallback(windowHandle, Input::MouseCallback);

		glfwShowWindow(windowHandle);

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

void Window::Draw()
{
	layout->Draw(windowHandle);
}