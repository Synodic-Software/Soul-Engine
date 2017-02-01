#include "RasterBackend.h"
#include "Vulkan/VulkanBackend.h"
#include "OpenGL/OpenGLBackend.h"

#include <memory>

namespace RasterBackend {

	Backend::Backend() {

	}

	Backend::~Backend() {

	}

	//encapsulate a backend and pass info to it
	namespace detail {
		std::unique_ptr<Backend> raster;
	}

	void Init() {

		/*if (glfwVulkanSupported() == GLFW_TRUE) {
			detail::raster.reset(new VulkanBackend());
		}
		else {*/
			detail::raster.reset(new OpenGLBackend());
		/*}*/

	}

	void SetWindowHints(GLFWwindow*& window) {
		detail::raster.get()->SetWindowHints(window);
	}

	void ResizeWindow(GLFWwindow* window, int x, int y) {
		detail::raster.get()->ResizeWindow(window, x, y);
	}

	void BuildWindow(GLFWwindow* window) {
		detail::raster.get()->BuildWindow(window);
	}

	void Terminate() {

	}

	void Draw(GLFWwindow* window) {
		detail::raster.get()->Draw(window);
	}
}
