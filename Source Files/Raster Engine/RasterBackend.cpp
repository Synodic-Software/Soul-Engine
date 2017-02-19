#include "RasterBackend.h"
#include "Vulkan/VulkanBackend.h"
#include "OpenGL/OpenGLBackend.h"
#include "Vulkan/VulkanShader.h"
#include "OpenGL/OpenGLShader.h"
#include <memory>

namespace RasterBackend {

	//backend
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

	Shader* CreateShader(const std::string& fileName, shader_t shaderT) {
		/*if (glfwVulkanSupported() == GLFW_TRUE) {
			return new VulkanShader(fileName, shaderT);
		}
		else {*/
			return new OpenGLShader(fileName, shaderT);
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
