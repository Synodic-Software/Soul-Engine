#include "RasterBackend.h"
#include "Vulkan/VulkanBackend.h"
#include "OpenGL/OpenGLBackend.h"
#include "Vulkan/VulkanShader.h"
#include "OpenGL/OpenGLShader.h"
#include "Vulkan/VulkanJob.h"
#include "OpenGL/OpenGLJob.h"
#include <memory>

namespace RasterBackend {

	//backend
	Backend::Backend() {

	}

	Backend::~Backend() {

	}

	namespace detail {
		std::unique_ptr<Backend> raster;
		BackendName backend;
	}

	void Init() {

		/*if (glfwVulkanSupported() == GLFW_TRUE) {
			detail::raster.reset(new VulkanBackend());
			backend = Vulkan;

		}
		else {*/
			detail::raster.reset(new OpenGLBackend());
			detail::backend = OpenGL;
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

	RasterJob* CreateJob() {
		/*if (glfwVulkanSupported() == GLFW_TRUE) {
		return new VulkanShader(fileName, shaderT);
		}
		else {*/
		return new OpenGLJob();
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

	void Draw(GLFWwindow* window, RasterJob* job) {
		detail::raster.get()->Draw(window, job);
	}
}
