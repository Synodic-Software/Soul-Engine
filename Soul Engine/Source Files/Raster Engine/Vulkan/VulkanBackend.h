#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Display\Window\Window.h"
#include "Raster Engine\RasterBackend.h"



class VulkanBackend : public RasterBackend::Backend {
public:
	VulkanBackend();
	~VulkanBackend();

	virtual void Init();
	virtual void BuildWindow(GLFWwindow*);
	virtual void PreRaster();
	virtual void PostRaster();
	virtual void Terminate();
	virtual void Draw();

private:

};