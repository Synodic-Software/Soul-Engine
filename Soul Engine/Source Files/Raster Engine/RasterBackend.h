#pragma once

#include "Display\Window\Window.h"

namespace RasterBackend {
	class Backend {
	public:
		Backend();
		~Backend();

		virtual void Init() = 0;
		virtual void SetWindowHints() = 0;
		virtual void BuildWindow(GLFWwindow*) = 0;
		virtual void PreRaster(GLFWwindow*) = 0;
		virtual void PostRaster(GLFWwindow*) = 0;
		virtual void Terminate() = 0;
		virtual void Draw(GLFWwindow*) = 0;
		virtual void ResizeWindow(GLFWwindow*, int, int) = 0;

	private:
	};

	//User: Do not touch
	namespace detail {
		extern Backend* raster;
	}

	void Init();

	//needs to be called from the main thread
	void SetWindowHints();

	void ResizeWindow(GLFWwindow*, int, int);

	void BuildWindow(GLFWwindow*);

	void PreRaster(GLFWwindow*);

	void PostRaster(GLFWwindow*);

	void Terminate();

	void Draw(GLFWwindow*);
}
