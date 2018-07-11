#include "VulkanAPI.h"

#include "GLFW/glfw3.h"

VulkanAPI::VulkanAPI():
	GraphicsAPI(RasterAPI::VULKAN)
{

	vk::ApplicationInfo applicationInfo;
	applicationInfo.apiVersion = VK_API_VERSION_1_1;
	applicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);//TODO forward the application version here
	applicationInfo.pApplicationName = "Soul Engine"; //TODO forward the application name here
	applicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0); //TODO forward the engine version here
	applicationInfo.pEngineName = "Soul Engine"; //TODO forward the engine name here

	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	vk::InstanceCreateInfo instanceCreationInfo;
	instanceCreationInfo.pApplicationInfo = &applicationInfo;
	instanceCreationInfo.enabledExtensionCount = glfwExtensionCount;
	instanceCreationInfo.ppEnabledExtensionNames = glfwExtensions;
	instanceCreationInfo.enabledLayerCount = 0;

	vulkanInstance_ = createInstanceUnique(instanceCreationInfo);

}
