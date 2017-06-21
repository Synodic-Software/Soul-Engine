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
		/* The raster */
		/* The raster */
		std::unique_ptr<RasterBase> raster;
	}

	/* The backend */
	/* The backend */
	BackendName backend;


	/* Initializes this object. */
	/* Initializes this object. */
	void Initialize() {

		/*if (glfwVulkanSupported() == GLFW_TRUE) {
			detail::raster.reset(new VulkanBackend());
			backend = Vulkan;

		}
		else {*/
			detail::raster.reset(new OpenGLBackend());
			backend = OpenGL;
		/*}*/

	}

	/* Makes context current. */
	/* Makes context current. */
	void MakeContextCurrent(){
		/*if (glfwVulkanSupported() == GLFW_TRUE) {
		return new VulkanShader(fileName, shaderT);
		}
		else {*/
		static_cast<OpenGLBackend*>(detail::raster.get())->MakeContextCurrent();
		/*}*/

	}

	/*
	 *    Creates a shader.
	 *
	 *    @param	fileName	Filename of the file.
	 *    @param	shaderT 	The shader t.
	 *
	 *    @return	Null if it fails, else the new shader.
	 */

	Shader* CreateShader(const std::string& fileName, shader_t shaderT) {
		/*if (glfwVulkanSupported() == GLFW_TRUE) {
			return new VulkanShader(fileName, shaderT);
		}
		else {*/
			return new OpenGLShader(fileName, shaderT);
		/*}*/
	}

	/*
	 *    Creates a buffer.
	 *
	 *    @param	size	The size.
	 *
	 *    @return	Null if it fails, else the new buffer.
	 */

	Buffer* CreateBuffer(uint size) {
		/*if (glfwVulkanSupported() == GLFW_TRUE) {
		return new VulkanBuffer();
		}
		else {*/
		return new OpenGLBuffer(size);
		/*}*/
	}

	/*
	 *    Creates the job.
	 *
	 *    @return	Null if it fails, else the new job.
	 */

	RasterJob* CreateJob() {
		/*if (glfwVulkanSupported() == GLFW_TRUE) {
		return new VulkanShader(fileName, shaderT);
		}
		else {*/
		return new OpenGLJob();
		/*}*/
	}

	/* Sets window hints. */
	/* Sets window hints. */
	void SetWindowHints() {
		detail::raster.get()->SetWindowHints();
	}

	/*
	 *    Resize window.
	 *
	 *    @param [in,out]	window	If non-null, the window.
	 *    @param 		 	x	  	The x coordinate.
	 *    @param 		 	y	  	The y coordinate.
	 */

	void ResizeWindow(GLFWwindow* window, int x, int y) {
		detail::raster.get()->ResizeWindow(window, x, y);
	}

	/*
	 *    Builds a window.
	 *
	 *    @param [in,out]	window	If non-null, the window.
	 */

	void BuildWindow(GLFWwindow* window) {
		detail::raster.get()->BuildWindow(window);
	}

	/* Terminates this object. */
	/* Terminates this object. */
	void Terminate() {

	}

	/*
	 *    Draws.
	 *
	 *    @param [in,out]	window	If non-null, the window.
	 *    @param [in,out]	job   	If non-null, the job.
	 */

	void Draw(GLFWwindow* window, RasterJob* job) {
		detail::raster.get()->Draw(window, job);
	}

	/*
	 *    Pre raster.
	 *
	 *    @param [in,out]	window	If non-null, the window.
	 */

	void PreRaster(GLFWwindow* window) {
		detail::raster.get()->PreRaster(window);
	}

	/*
	 *    Posts a raster.
	 *
	 *    @param [in,out]	window	If non-null, the window.
	 */

	void PostRaster(GLFWwindow* window) {
		detail::raster.get()->PostRaster(window);
	}
}
