#pragma once

#include "Display\Window\AbstractWindow.h"
#include "Raster Engine/Graphics API/GraphicsAPI.h"

class VulkanAPI : public GraphicsAPI {
public:

	VulkanAPI();
	~VulkanAPI();

};