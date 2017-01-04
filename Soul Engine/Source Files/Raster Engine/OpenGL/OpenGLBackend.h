#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Display\Window\Window.h"
#include "Raster Engine\RasterBackend.h"


class OpenGLBackend : public RasterBackend::Backend {
public:
	OpenGLBackend();
	~OpenGLBackend();

	virtual void Init();
	virtual void SCreateWindow(Window*);
	virtual void PreRaster();
	virtual void PostRaster();
	virtual void Terminate();
	virtual void Draw();

private:

};