#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Display\Window\Window.h"
#include "Raster Engine\RasterBackend.h"


class OpenGLBackend : public RasterBackend::Backend {
public:
	OpenGLBackend();
	~OpenGLBackend();

	virtual void Init();
	virtual void CreateWindow(Window*, GLFWmonitor*, GLFWwindow*);
private:

};