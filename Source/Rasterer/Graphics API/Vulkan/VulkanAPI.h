#pragma once

//#include "Display\Window\AbstractWindow.h"
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