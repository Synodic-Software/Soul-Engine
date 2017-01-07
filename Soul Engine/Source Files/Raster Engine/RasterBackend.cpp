#include "RasterBackend.h"
#include "Vulkan/VulkanBackend.h"
#include "OpenGL/OpenGLBackend.h"

namespace RasterBackend {

	Backend::Backend() {

	}

	Backend::~Backend() {

	}

	//encapsulate a backend and pass info to it
	namespace detail {
		Backend* raster;
		VulkanBackend vBack;
		OpenGLBackend oBack;
	}

	void Init() {

		if (glfwVulkanSupported() == GLFW_TRUE) {
			detail::raster = &detail::vBack;
		}
		else {
			detail::raster = &detail::oBack;
		}

		detail::raster->Init();

	}

	void SetWindowHints() {
		detail::raster->SetWindowHints();
	}

	void ResizeWindow(GLFWwindow* window, int x, int y) {
		detail::raster->ResizeWindow(window, x, y);
	}

	void BuildWindow(GLFWwindow* window) {
		detail::raster->BuildWindow(window);
	}

	void PreRaster(GLFWwindow* window) {
		detail::raster->PreRaster(window);
	}

	void PostRaster(GLFWwindow* window) {
		detail::raster->PostRaster(window);
	}

	void Terminate() {
		detail::raster->Terminate();
	}

	void Draw(GLFWwindow* window) {
		detail::raster->Draw(window);
	}
}
