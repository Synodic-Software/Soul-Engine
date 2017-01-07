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

	void BuildWindow(GLFWwindow* window) {
		detail::raster->BuildWindow(window);
	}

	void PreRaster() {
		detail::raster->PreRaster();
	}

	void PostRaster() {
		detail::raster->PostRaster();
	}

	void Terminate() {
		detail::raster->Terminate();
	}

	void Draw() {
		detail::raster->Draw();
	}
}
