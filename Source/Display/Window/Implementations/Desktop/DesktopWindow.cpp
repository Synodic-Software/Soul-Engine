#include "DesktopWindow.h"
#include "Display\Window\Implementations\Desktop\DesktopManager.h"

#include "Utility\Logger.h"
#include "Raster Engine\RasterManager.h"
#include "Parallelism\Scheduler.h"
#include "Input\InputManager.h"

DesktopWindow::DesktopWindow(WindowType inWin, const std::string& inTitle, uint x, uint y, uint iwidth, uint iheight, void* monitorIn, void* sharedContext) :
AbstractWindow(inWin, inTitle, x,y, iwidth, iheight, monitorIn, sharedContext)
{
	glfwWindowHint(GLFW_SAMPLES, GLFW_DONT_CARE);
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

	GLFWmonitor* monitor = static_cast<GLFWmonitor*>(monitorIn);
	GLFWwindow* context = static_cast<GLFWwindow*>(sharedContext);

	const GLFWvidmode* mode = glfwGetVideoMode(monitor);

	glfwWindowHint(GLFW_RED_BITS, mode->redBits);
	glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
	glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
	glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

	if (windowType == FULLSCREEN) {

		glfwWindowHint(GLFW_RESIZABLE, false);
		glfwWindowHint(GLFW_DECORATED, false);

		windowHandle = glfwCreateWindow(width, height, title.c_str(), monitor, context);

	}
	else if (windowType == WINDOWED) {

		glfwWindowHint(GLFW_RESIZABLE, true);

		windowHandle = glfwCreateWindow(width, height, title.c_str(), nullptr, context);

	}
	else if (windowType == BORDERLESS) {

		glfwWindowHint(GLFW_RESIZABLE, false);
		glfwWindowHint(GLFW_DECORATED, false);

		windowHandle = glfwCreateWindow(width, height, title.c_str(), nullptr, context);

	}
	else {
		glfwWindowHint(GLFW_RESIZABLE, false);
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

		windowHandle = glfwCreateWindow(width, height, title.c_str(), nullptr, context);

	}
	

	// convert windowHandle from void* to a GLFWwindow*
	GLFWwindow* winHandle = static_cast<GLFWwindow*> (windowHandle);


	if (winHandle == nullptr)
	{
		S_LOG_FATAL("Could not Create GLFW Window");
	}

	//the backend is the new user

	DesktopWindow* thisWindow = this;


	glfwSetInputMode(winHandle, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
	glfwSetWindowUserPointer(winHandle, thisWindow);

	//all window related callbacks
	glfwSetWindowSizeCallback(winHandle, [](GLFWwindow* w, int x, int y)
	{
		DesktopManager::Instance().Resize(w, x, y);
	});

	glfwSetWindowPosCallback(winHandle, [](GLFWwindow* w, int x, int y)
	{
		DesktopManager::Instance().WindowPos(w, x, y);
	});

	glfwSetWindowRefreshCallback(winHandle, [](GLFWwindow* w)
	{
		DesktopManager::Instance().Refresh(w);
	});

	glfwSetWindowCloseCallback(winHandle, [](GLFWwindow* w)
	{
		DesktopManager::Instance().Close(w);
	});

	glfwShowWindow(winHandle);

	InputManager::AttachWindow(winHandle);

}

/* Destructor. */
DesktopWindow::~DesktopWindow()
{
	if (windowHandle) {
		glfwDestroyWindow(static_cast<GLFWwindow*>(windowHandle));
	}
}

/* Draws this object. */
void DesktopWindow::Draw()
{
	layout->Draw();
}