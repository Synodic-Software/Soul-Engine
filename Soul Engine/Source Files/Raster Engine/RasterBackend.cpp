#include "RasterBackend.h"
#include "Vulkan/VulkanBackend.h"
#include "OpenGL/OpenGLBackend.h"

static VulkanBackend vBack;
static OpenGLBackend oBack;

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
			detail::raster = &vBack;
		}
		else {
			detail::raster = &oBack;
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
