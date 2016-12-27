#include "Window.h"
#include "Utility\Logger.h"
#include "Input\Input.h"
#include "Raster Engine\RasterBackend.h"

Window::Window(const std::string& inTitle, uint x, uint y, uint iwidth, uint iheight, GLFWmonitor* monitorIn)
: title(inTitle){

	xPos = x;
	yPos = y;
	width = iwidth;
	height = iheight;

	RasterBackend::CreateWindow(this, monitorIn,nullptr);
	//glfwSetWindowUserPointer(windowOut, &VulkanBackend::GetInstance());
	//glfwSetWindowSizeCallback(windowOut, VulkanBackend::OnWindowResized);

	//////////////////////////

	/*Soul::renderer rend = {
	new Renderer(glm::uvec2(int(xSize*mode->width), int(ySize*mode->height))),
	SPECTRAL,
	1.0f
	};

	Soul::renderObjects.push_back(rend);*/

	//////////////////////////

	//VulkanBackend::GetInstance().AddWindow(windowOut, int(xSize*mode->width), int(ySize*mode->height));

}


Window::~Window()
{
}

void Window::Draw() {
	//int width, height;
	//glfwGetWindowSize(masterWindow, &width, &height);
	//glfwSetCursorPos(masterWindow, width / 2.0f, height / 2.0f);
//	VulkanBackend::GetInstance().DrawFrame(masterWindow, width, height);
}