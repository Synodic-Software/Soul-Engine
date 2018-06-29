#pragma once

//#include "Display\Window\SoulWindow.h"
#include "Rasterer/Graphics API/GraphicsAPI.h"
#include <vulkan/vulkan.h>

class VulkanAPI : public GraphicsAPI {
public:

	VulkanAPI();
	~VulkanAPI();
private:
	void InitInstance();
	void DeInstance();
	VkInstance       _instance = nullptr;
 }; 