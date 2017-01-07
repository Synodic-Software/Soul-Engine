#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Display\Window\Window.h"
#include "Raster Engine\RasterBackend.h"

class OpenGLBackend : public RasterBackend::Backend {
public:
	OpenGLBackend();
	~OpenGLBackend();

	virtual void Init();
	virtual	void SetWindowHints();
	virtual void ResizeWindow(GLFWwindow*, int, int);
	virtual void BuildWindow(GLFWwindow*);
	virtual void PreRaster(GLFWwindow*);
	virtual void PostRaster(GLFWwindow*);
	virtual void Terminate();
	virtual void Draw(GLFWwindow*);

	struct WindowInformation
	{
		GLFWwindow* window;
		GLEWContext* glContext;
		unsigned int ID;
	};

private:

	void MakeContextCurrent(WindowInformation*);
};