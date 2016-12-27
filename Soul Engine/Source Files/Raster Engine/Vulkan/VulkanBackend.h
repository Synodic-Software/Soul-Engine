#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Display\Window\Window.h"
#include "Raster Engine\RasterBackend.h"



class VulkanBackend : public RasterBackend::Backend {
public:
	VulkanBackend();
	~VulkanBackend();

	virtual void Init();
	virtual void CreateWindow(Window*, GLFWmonitor*, GLFWwindow*);
private:

};