#include "RasterBackend.h"
#include "Vulkan/VulkanBackend.h"
#include "OpenGL/OpenGLBackend.h"


namespace RasterBackend {

	Backend::Backend() {

	}

	Backend::~Backend() {

	}

	namespace detail {
		Backend* raster;
	}

	void Init() {
		if (glfwVulkanSupported() == GLFW_TRUE) {
			detail::raster = new VulkanBackend();
		}
		else {
			detail::raster = new OpenGLBackend();
		}


		detail::raster->Init();
	}

	void CreateWindow(Window& window) {
		detail::raster->CreateWindow(window);
	}

}
