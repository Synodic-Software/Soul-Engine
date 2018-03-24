#include "GLFW.h"

#include "Utility/Logger.h"
#include "Raster Engine/RasterManager.h"
#include "Multithreading/Scheduler.h"
#include "Display/Window/WindowManager.h"
#include "Input/InputManager.h"

GLFW::GLFW(WindowType inWin, const std::string& inTitle, uint x, uint y, uint iwidth, uint iheight, GLFWmonitor* monitorIn, GLFWwindow* sharedContext) :
	Window(inTitle, x, y, iwidth, iheight)
{
	windowType = inWin;
	windowHandle = nullptr;

	glfwWindowHint(GLFW_SAMPLES, GLFW_DONT_CARE);
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

	const GLFWvidmode* monitorType = glfwGetVideoMode(monitorIn);

	glfwWindowHint(GLFW_RED_BITS, monitorType->redBits);
	glfwWindowHint(GLFW_GREEN_BITS, monitorType->greenBits);
	glfwWindowHint(GLFW_BLUE_BITS, monitorType->blueBits);
	glfwWindowHint(GLFW_REFRESH_RATE, monitorType->refreshRate);

	if (windowType == FULLSCREEN) {

		glfwWindowHint(GLFW_RESIZABLE, false);
		glfwWindowHint(GLFW_DECORATED, false);

		windowHandle = glfwCreateWindow(width, height, title.c_str(), monitorIn, sharedContext);

	} else if (windowType == WINDOWED) {

		glfwWindowHint(GLFW_RESIZABLE, true);

		windowHandle = glfwCreateWindow(width, height, title.c_str(), nullptr, sharedContext);

	} else if (windowType == BORDERLESS) {

		glfwWindowHint(GLFW_RESIZABLE, false);
		glfwWindowHint(GLFW_DECORATED, false);

		windowHandle = glfwCreateWindow(width, height, title.c_str(), nullptr, sharedContext);

	} else {
		glfwWindowHint(GLFW_RESIZABLE, false);
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

		windowHandle = glfwCreateWindow(width, height, title.c_str(), nullptr, sharedContext);
	}

	if (windowHandle == nullptr) {
		S_LOG_FATAL("Could not Create GLFW Window");
	}

	//the backend is the new user

	Window* thisWindow = this;

	glfwSetInputMode(windowHandle, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
	glfwSetWindowUserPointer(windowHandle, thisWindow);

	//all window related callbacks
	glfwSetWindowSizeCallback(windowHandle, [](GLFWwindow* w, int x, int y) {
		WindowManager::Instance().Resize(w, x, y);
	});

	glfwSetWindowPosCallback(windowHandle, [](GLFWwindow* w, int x, int y) {
		WindowManager::Instance().WindowPos(w, x, y);
	});

	glfwSetWindowRefreshCallback(windowHandle, [](GLFWwindow* w) {
		WindowManager::Instance().Refresh(w);
	});

	glfwSetWindowCloseCallback(windowHandle, [](GLFWwindow* w) {
		WindowManager::Instance().Close(w);
	});

	glfwShowWindow(windowHandle);

	InputManager::AttachWindow(windowHandle);
}

/* Destructor. */
GLFW::~GLFW() {
	if (windowHandle) {
		glfwDestroyWindow(windowHandle);
	}
}

/* Draws this object. */
void GLFW::Draw() {
	layout->Draw();
}
