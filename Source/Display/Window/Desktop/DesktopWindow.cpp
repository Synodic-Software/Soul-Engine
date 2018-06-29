#include "DesktopWindow.h"

#include "Core/Utility/Log/Logger.h"
#include "Parallelism/Fiber/Scheduler.h"
#include "Transput/Input/InputManager.h"

DesktopWindow::DesktopWindow(WindowParameters& params, void* monitorIn, void* sharedContext) :
SoulWindow(params)
{
	glfwWindowHint(GLFW_SAMPLES, GLFW_DONT_CARE);
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

	//TODO: Rework Monitor Pointer
	GLFWmonitor* monitor = static_cast<GLFWmonitor*>(monitorIn);
	GLFWwindow* context = static_cast<GLFWwindow*>(sharedContext);
	
	if (monitor == nullptr) {
		monitor = glfwGetPrimaryMonitor();
	}

	const GLFWvidmode* mode = glfwGetVideoMode(monitor);

	glfwWindowHint(GLFW_RED_BITS, mode->redBits);
	glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
	glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
	glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

	if (windowParams_.type == WindowType::FULLSCREEN) {

		glfwWindowHint(GLFW_RESIZABLE, false);
		glfwWindowHint(GLFW_DECORATED, false);

		context_ = glfwCreateWindow(windowParams_.pixelWidth, windowParams_.pixelHeight, windowParams_.title.c_str(), monitor, context);

	}
	else if (windowParams_.type == WindowType::WINDOWED) {

		glfwWindowHint(GLFW_RESIZABLE, true);

		context_ = glfwCreateWindow(windowParams_.pixelWidth, windowParams_.pixelHeight, windowParams_.title.c_str(), nullptr, context);

	}
	else if (windowParams_.type == WindowType::BORDERLESS) {

		glfwWindowHint(GLFW_RESIZABLE, false);
		glfwWindowHint(GLFW_DECORATED, false);

		context_ = glfwCreateWindow(windowParams_.pixelWidth, windowParams_.pixelHeight, windowParams_.title.c_str(), nullptr, context);

	}
	else {
		glfwWindowHint(GLFW_RESIZABLE, false);
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

		context_ = glfwCreateWindow(windowParams_.pixelWidth, windowParams_.pixelHeight, windowParams_.title.c_str(), nullptr, context);

	}
	

	// convert windowHandle from void* to a GLFWwindow*
	auto winHandle = std::any_cast<GLFWwindow*>(context_);


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
		auto thisWindow = static_cast<DesktopWindow*>(glfwGetWindowUserPointer(w));
		thisWindow->Resize(x, y);
	});

	glfwSetWindowPosCallback(winHandle, [](GLFWwindow* w, int x, int y)
	{
		auto thisWindow = static_cast<DesktopWindow*>(glfwGetWindowUserPointer(w));
		thisWindow->PositionUpdate(x, y);
	});

	glfwSetWindowRefreshCallback(winHandle, [](GLFWwindow* w)
	{
		auto thisWindow = static_cast<DesktopWindow*>(glfwGetWindowUserPointer(w));
		thisWindow->Refresh();
	});

	glfwSetWindowCloseCallback(winHandle, [](GLFWwindow* w)
	{
		auto thisWindow = static_cast<DesktopWindow*>(glfwGetWindowUserPointer(w));
		thisWindow->Close();
	});

	glfwShowWindow(winHandle);

	InputManager::AttachWindow(winHandle);

}

/* Destructor. */
DesktopWindow::~DesktopWindow()
{
	if (context_.has_value()) {
		glfwDestroyWindow(std::any_cast<GLFWwindow*>(context_));
	}
}

void DesktopWindow::Draw()
{
	layout_->Draw();
}


void DesktopWindow::Refresh() {
	
}

void DesktopWindow::Close() {

}

void DesktopWindow::Resize(int, int) {

}

void DesktopWindow::PositionUpdate(int, int) {

}

void DesktopWindow::SetLayout(Layout* layout) {
	layout_.reset(layout);
	//layout_->UpdateWindow(static_cast<GLFWwindow*>(windows.back().get()->windowHandle));
	layout_->UpdatePositioning(glm::uvec2(windowParams_.pixelPosX, windowParams_.pixelPosY), glm::uvec2(windowParams_.pixelWidth, windowParams_.pixelHeight));
	layout_->RecreateData();
}