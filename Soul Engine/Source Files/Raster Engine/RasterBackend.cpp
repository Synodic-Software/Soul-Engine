#include "RasterBackend.h"
#include "Vulkan/VulkanBackend.h"
#include "OpenGL/OpenGLBackend.h"


namespace RasterBackend {

	Backend::Backend() {

	}

	Backend::~Backend() {

	}

	namespace detail {

		VulkanBackend vBack;
		OpenGLBackend oBack;

		Backend* raster;
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

	void CreateWindow(Window* window, GLFWmonitor* moniter, GLFWwindow* share) {
		detail::raster->CreateWindow(window, moniter, share);
	}

}
