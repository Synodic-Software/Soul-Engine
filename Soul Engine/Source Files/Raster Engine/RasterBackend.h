#pragma once

#include "Display\Window\Window.h"

namespace RasterBackend {
	class Backend {
	public:
		Backend();
		~Backend();

		virtual void Init() = 0;
		virtual void BuildWindow(GLFWwindow*) = 0;
		virtual void PreRaster() = 0;
		virtual void PostRaster() = 0;
		virtual void Terminate() = 0;
		virtual void Draw() = 0;

	private:
	};

	//User: Do not touch
	namespace detail {
		extern Backend* raster;
	}

	void Init();

	void BuildWindow(GLFWwindow*);

	void PreRaster();

	void PostRaster();

	void Terminate();

	void Draw();
}
