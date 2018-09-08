#include "DesktopWindow.h"

#include "Core/Utility/Log/Logger.h"
#include "Parallelism/Fiber/Scheduler.h"
#include "Transput/Input/InputManager.h"
#include "Rasterer/Graphics API/Vulkan/VulkanSwapChain.h"

DesktopWindow::DesktopWindow(EntityManager* entityManager, WindowParameters& params, GLFWmonitor* monitor, DesktopInputManager& inputManager, RasterManager& rasterManager) :
	Window(params),
	inputManager_(&inputManager),
	rasterManager_(&rasterManager)
{

	//TODO: Rework Monitor Pointer
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

	GLFWwindow* context = glfwCreateWindow(windowParams_.pixelSize.x, windowParams_.pixelSize.y, windowParams_.title.c_str(), fullscreenMonitor, nullptr);

	assert(context);
	context_ = context;


	//context related settings
	glfwSetInputMode(context, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

	//set so the window object that holds the context is visible in callbacks
	glfwSetWindowUserPointer(context, this);

	//all window related callbacks
	glfwSetWindowSizeCallback(context, [](GLFWwindow* w, int x, int y)
	{
		const auto thisWindow = static_cast<DesktopWindow*>(glfwGetWindowUserPointer(w));
		thisWindow->Resize(x, y);
	});

	glfwSetWindowPosCallback(context, [](GLFWwindow* w, int x, int y)
	{
		const auto thisWindow = static_cast<DesktopWindow*>(glfwGetWindowUserPointer(w));
		thisWindow->PositionUpdate(x, y);
	});

	glfwSetWindowRefreshCallback(context, [](GLFWwindow* w)
	{
		const auto thisWindow = static_cast<DesktopWindow*>(glfwGetWindowUserPointer(w));
		thisWindow->Refresh();
	});

	glfwSetFramebufferSizeCallback(context, [](GLFWwindow* w, int x, int y)
	{
		const auto thisWindow = static_cast<DesktopWindow*>(glfwGetWindowUserPointer(w));
		thisWindow->FrameBufferResize(x, y);
	});

	glfwSetWindowCloseCallback(context, [](GLFWwindow* w)
	{
		const auto thisWindow = static_cast<DesktopWindow*>(glfwGetWindowUserPointer(w));
		thisWindow->Close();
	});

	//create surface
	surface_ = rasterManager.CreateSurface(context_);

	//create device //TODO: move deviceManager
	device_ = rasterManager.CreateDevice(surface_);

	//create swapchain
	swapChain_ = rasterManager.CreateSwapChain(device_, surface_, windowParams_.pixelSize);


	//only show the window once all proper callbacks and settings are in place
	glfwShowWindow(context);

}

void DesktopWindow::Terminate() {

	if (context_.has_value()) {
		glfwDestroyWindow(std::any_cast<GLFWwindow*>(context_));
	}

}

void DesktopWindow::Draw()
{
	//entity
	rasterManager_->Raster(swapChain_);

}


void DesktopWindow::Refresh() {



}

void DesktopWindow::Close() {

}

void DesktopWindow::Resize(int x, int y) {

}

void DesktopWindow::FrameBufferResize(int x, int y) {

	rasterManager_->ResizeSwapChain(swapChain_, x, y);

}


void DesktopWindow::PositionUpdate(int, int) {

}

void DesktopWindow::SetLayout(Layout* layout) {
	/*layout_.reset(layout);
	layout_->UpdateWindow(static_cast<GLFWwindow*>(windows.back().get()->windowHandle));
	layout_->UpdatePositioning(glm::uvec2(windowParams_.pixelPosX, windowParams_.pixelPosY), glm::uvec2(windowParams_.pixelWidth, windowParams_.pixelHeight));
	layout_->RecreateData();*/
}

DesktopInputManager& DesktopWindow::GetInputSet() const {
	return *inputManager_;
}