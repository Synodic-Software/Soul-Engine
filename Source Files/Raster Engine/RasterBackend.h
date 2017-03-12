#pragma once

#include "Display\Window\Window.h"
#include "Raster Engine\Shader.h"
#include "Raster Engine\RasterJob.h"
#include "Raster Engine\Buffer.h"

enum BackendName { OpenGL, Vulkan};

namespace RasterBackend {

		/////////////////////////////////////////
		/*         Backend Definitions         */
		/////////////////////////////////////////

	class Backend {
	public:
		Backend();
		~Backend();

		virtual void SetWindowHints(GLFWwindow*&) = 0;
		virtual void BuildWindow(GLFWwindow*) = 0;
		virtual void Draw(GLFWwindow*,RasterJob*) = 0;
		virtual void ResizeWindow(GLFWwindow*, int, int) = 0;

		template<typename Fn,
			typename ... Args>
			void RasterFunction( Fn && fn, Args && ... args) = 0;

	private:
	};

	namespace detail {
		extern std::unique_ptr<Backend> raster;
	}
	

		/////////////////////////////////////////
		/*         Public Definitions         */
		/////////////////////////////////////////

	extern BackendName backend;


	void Init();

	//needs to be called from the main thread
	void SetWindowHints(GLFWwindow*&);

	Shader* CreateShader(const std::string&, shader_t);

	Buffer* CreateBuffer(uint size);

	RasterJob* CreateJob();

	template<typename Fn,
		typename ... Args>
		void RasterFunction( Fn && fn, Args && ... args) {
		/*if (glfwVulkanSupported() == GLFW_TRUE) {
		return new VulkanShader(fileName, shaderT);
		}
		else {*/
		static_cast<OpenGLBackend*>(detail::raster.get())->RasterFunction(fn, std::forward<Args>(args)...);
		/*}*/

	}

	void ResizeWindow(GLFWwindow*, int, int);

	void BuildWindow(GLFWwindow*);

	void Terminate();

	void Draw(GLFWwindow*,RasterJob*);
}
