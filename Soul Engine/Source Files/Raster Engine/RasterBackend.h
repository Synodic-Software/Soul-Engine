#pragma once

#include "Display\Window\Window.h"

namespace RasterBackend {
	class Backend {
	public:
		Backend();
		~Backend();

		virtual void Init() = 0;
		virtual void CreateWindow(Window&) = 0;
	private:
	};

	//User: Do not touch
	namespace detail {
		extern Backend* raster;
	}

	void Init();

	void CreateWindow(Window&);
}
