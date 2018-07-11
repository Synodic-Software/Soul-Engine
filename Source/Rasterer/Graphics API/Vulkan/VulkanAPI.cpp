#include "VulkanAPI.h"

#include "GLFW/glfw3.h"

VulkanAPI::VulkanAPI():
	GraphicsAPI(RasterAPI::VULKAN)
{

	glfwInit();

	vk::ApplicationInfo applicationInfo;
	applicationInfo.apiVersion = VK_API_VERSION_1_1;
	applicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	applicationInfo.pApplicationName = "Soul Engine";
	applicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	applicationInfo.pEngineName = "Soul Engine";

	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	vk::InstanceCreateInfo instanceCreationInfo;
	instanceCreationInfo.pApplicationInfo = &applicationInfo;
	instanceCreationInfo.enabledExtensionCount = glfwExtensionCount;
	instanceCreationInfo.ppEnabledExtensionNames = glfwExtensions;
	instanceCreationInfo.enabledLayerCount = 0;

	vulkanInstance_ = createInstanceUnique(instanceCreationInfo);

}
