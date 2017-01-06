#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Display\Window\Window.h"
#include "Raster Engine\RasterBackend.h"

#include <vector>


class OpenGLBackend : public RasterBackend::Backend {
public:
	OpenGLBackend();
	~OpenGLBackend();

	virtual void Init();
	virtual void BuildWindow(GLFWwindow*);
	virtual void PreRaster();
	virtual void PostRaster();
	virtual void Terminate();
	virtual void Draw();

	struct WindowInformation
	{
		GLFWwindow* window;
		GLEWContext* glContext;
		unsigned int ID;
	};

private:

	void MakeContextCurrent(WindowInformation*);
};