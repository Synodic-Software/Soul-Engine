#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Display\Window\Window.h"
#include "Raster Engine\RasterBackend.h"



class VulkanBackend : public RasterBackend::Backend {
public:
	VulkanBackend();
	~VulkanBackend();

	virtual void Init();
	virtual void SetWindowHints();
	virtual void ResizeWindow(GLFWwindow*, int, int);
	virtual void BuildWindow(GLFWwindow*);
	virtual void PreRaster(GLFWwindow*);
	virtual void PostRaster(GLFWwindow*);
	virtual void Terminate();
	virtual void Draw(GLFWwindow*);

private:

};