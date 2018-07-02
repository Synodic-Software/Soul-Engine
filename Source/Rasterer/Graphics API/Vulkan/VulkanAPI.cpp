#include "VulkanAPI.h"

#include <GLFW\glfw3.h>

VulkanAPI::VulkanAPI(): GraphicsAPI::GraphicsAPI()
{
	backendType = Vulkan;
	InitInstance();
}

VulkanAPI::~VulkanAPI() {
	DeInstance();
}

void VulkanAPI::InitInstance()
{
	VkApplicationInfo application_info{};
	application_info.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	application_info.apiVersion = VK_API_VERSION_1_0;
	application_info.applicationVersion = VK_MAKE_VERSION(1,0, 0);
	application_info.pApplicationName = "Soul Engine Test";
	application_info.pEngineName = "Soul Engine";
	application_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	
		
		
	VkInstanceCreateInfo instance_create_info{};
	instance_create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instance_create_info.pApplicationInfo = &application_info;

	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions;

	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	instance_create_info.enabledExtensionCount = glfwExtensionCount;
	instance_create_info.ppEnabledExtensionNames = glfwExtensions;
	instance_create_info.enabledLayerCount = 0;

	VkResult result = vkCreateInstance(&instance_create_info, nullptr, &_instance);
	if (vkCreateInstance(&instance_create_info, nullptr, &_instance) != VK_SUCCESS) {
		//throw std::runtime_error("failed to create instance!");
	}
	
}

void VulkanAPI::DeInstance()
{
	vkDestroyInstance(_instance, nullptr);
	_instance = nullptr;
}
