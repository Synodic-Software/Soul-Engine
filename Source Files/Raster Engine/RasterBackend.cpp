#include "RasterBackend.h"
#include "Vulkan/VulkanBackend.h"
#include "OpenGL/OpenGLBackend.h"
#include "Vulkan/VulkanShader.h"
#include "OpenGL/OpenGLShader.h"
#include "Vulkan/VulkanJob.h"
#include "OpenGL/OpenGLJob.h"
#include "Vulkan/VulkanBuffer.h"
#include "OpenGL/OpenGLBuffer.h"
#include <memory>

namespace RasterBackend {

	namespace detail {
		std::unique_ptr<RasterBase> raster;
	}

	BackendName backend;


	void Init() {

		/*if (glfwVulkanSupported() == GLFW_TRUE) {
			detail::raster.reset(new VulkanBackend());
			backend = Vulkan;

		}
		else {*/
			detail::raster.reset(new OpenGLBackend());
			backend = OpenGL;
		/*}*/

	}

	void MakeContextCurrent(){
		/*if (glfwVulkanSupported() == GLFW_TRUE) {
		return new VulkanShader(fileName, shaderT);
		}
		else {*/
		static_cast<OpenGLBackend*>(detail::raster.get())->MakeContextCurrent();
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

	Buffer* CreateBuffer(uint size) {
		/*if (glfwVulkanSupported() == GLFW_TRUE) {
		return new VulkanBuffer();
		}
		else {*/
		return new OpenGLBuffer(size);
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

	void SetWindowHints() {
		detail::raster.get()->SetWindowHints();
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
