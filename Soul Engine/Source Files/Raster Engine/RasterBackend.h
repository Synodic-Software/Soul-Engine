#pragma once

#include "Display\Window\Window.h"

namespace RasterBackend {
	class Backend {
	public:
		Backend();
		~Backend();

		virtual void SetWindowHints() = 0;
		virtual void BuildWindow(GLFWwindow*) = 0;
		virtual void Draw(GLFWwindow*) = 0;
		virtual void ResizeWindow(GLFWwindow*, int, int) = 0;

	private:
	};

	void Init();

	//needs to be called from the main thread
	void SetWindowHints();

	void ResizeWindow(GLFWwindow*, int, int);

	void BuildWindow(GLFWwindow*);

	void Terminate();

	void Draw(GLFWwindow*);
}
