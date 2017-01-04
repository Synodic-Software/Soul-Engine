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

	void SCreateWindow(Window* window) {
		detail::raster->SCreateWindow(window);
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
