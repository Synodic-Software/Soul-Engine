#pragma once

#include "Display\Window\Window.h"
#include "Raster Engine\Shader.h"

namespace RasterBackend {
	class Backend {
	public:
		Backend();
		~Backend();

		virtual void SetWindowHints(GLFWwindow*&) = 0;
		virtual void BuildWindow(GLFWwindow*) = 0;
		virtual void Draw(GLFWwindow*) = 0;
		virtual void ResizeWindow(GLFWwindow*, int, int) = 0;


	private:
	};

	

	void Init();

	//needs to be called from the main thread
	void SetWindowHints(GLFWwindow*&);

	Shader* CreateShader(const std::string&, shader_t);

	RasterJob* CreateJob();

	void ResizeWindow(GLFWwindow*, int, int);

	void BuildWindow(GLFWwindow*);

	void Terminate();

	void Draw(GLFWwindow*);
}
