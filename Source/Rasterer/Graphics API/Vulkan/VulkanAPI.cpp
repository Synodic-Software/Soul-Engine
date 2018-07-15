#include "VulkanAPI.h"

#include "GLFW/glfw3.h"
#include "VulkanSwapChain.h"


VulkanAPI::VulkanAPI() :
	GraphicsAPI(RasterAPI::VULKAN)
{

	//setup Vulkan app info
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

	vulkanInstance_ = std::make_shared<vk::Instance>(createInstance(
		instanceCreationInfo
	));

	//setup devices
	//TODO: abstract devices
	physicalDevices_ = vulkanInstance_->enumeratePhysicalDevices();

	//TODO should never trigger assert
	assert(!physicalDevices.empty());

	//TODO minimize memory/runtime impact
	//required extensions
	requiredExtensions_.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

}

VulkanAPI::~VulkanAPI() {

	for (const auto& logicalDevice : logicalDevices_) {
		logicalDevice.destroy();
	}

	vulkanInstance_->destroy();

}

std::unique_ptr<SwapChain> VulkanAPI::CreateSwapChain(std::any& windowContext, glm::uvec2& size) {
	return std::make_unique<VulkanSwapChain>(vulkanInstance_, physicalDevices_, requiredExtensions_, windowContext, size);
}
