#include "VulkanContext.h"

#include "GLFW/glfw3.h"

#include "VulkanSwapChain.h"
#include "VulkanSurface.h"
#include "Parallelism/Fiber/Scheduler.h"

#include <set>

//TODO cleanup global
const std::vector<const char*> validationLayers = {
	"VK_LAYER_LUNARG_assistant_layer",
	"VK_LAYER_LUNARG_standard_validation"
};

VulkanContext::VulkanContext(Scheduler& scheduler, EntityManager& entityManger) :
	RasterContext(RasterAPI::VULKAN),
	scheduler_(scheduler),
	entityManager_(entityManger)
{

	//setup Vulkan app info
	vk::ApplicationInfo applicationInfo;
	applicationInfo.apiVersion = VK_API_VERSION_1_1;
	applicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);//TODO forward the application version here
	applicationInfo.pApplicationName = "Soul Engine"; //TODO forward the application name here
	applicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0); //TODO forward the engine version here
	applicationInfo.pEngineName = "Soul Engine"; //TODO forward the engine name here

	//set device extensions
	requiredDeviceExtensions_.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

	//set instance extensions

	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	for (uint i = 0; i < glfwExtensionCount; ++i) {
		requiredInstanceExtensions_.push_back(glfwExtensions[i]);
	}

	//TODO minimize memory/runtime impact
	if constexpr (validationEnabled_) {
		requiredInstanceExtensions_.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}


	vk::InstanceCreateInfo instanceCreationInfo;
	instanceCreationInfo.pApplicationInfo = &applicationInfo;
	instanceCreationInfo.enabledExtensionCount = static_cast<uint32_t>(requiredInstanceExtensions_.size());
	instanceCreationInfo.ppEnabledExtensionNames = requiredInstanceExtensions_.data();

	if constexpr (validationEnabled_) {
		instanceCreationInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		instanceCreationInfo.ppEnabledLayerNames = validationLayers.data();
	}
	else {
		instanceCreationInfo.enabledLayerCount = 0;
	}

	instance_ = createInstance(instanceCreationInfo);

	//setup devices
	//TODO: abstract physical devices
	//TODO: garuntee size constness
	physicalDevices_ = instance_.enumeratePhysicalDevices();

	//TODO should never trigger assert
	assert(!physicalDevices_.empty());

	//create debugging callback
	if constexpr (validationEnabled_) {

		dispatcher_ = vk::DispatchLoaderDynamic(instance_);

		vk::DebugUtilsMessengerCreateInfoEXT messangerCreateInfo;
		messangerCreateInfo.flags = vk::DebugUtilsMessengerCreateFlagBitsEXT(0);
		messangerCreateInfo.messageSeverity =
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
		messangerCreateInfo.messageType =
			vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
			vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
			vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
		messangerCreateInfo.pfnUserCallback = DebugCallback;
		messangerCreateInfo.pUserData = nullptr;

		debugMessenger_ = instance_.createDebugUtilsMessengerEXT(messangerCreateInfo, nullptr, dispatcher_);

	}
}

VulkanContext::~VulkanContext() {

	if constexpr (validationEnabled_) {

		instance_.destroyDebugUtilsMessengerEXT(debugMessenger_, nullptr, dispatcher_);

	}

	entityManager_.RemoveComponent<VulkanDevice>();
	entityManager_.RemoveComponent<VulkanSurface>();

	instance_.destroy();

}

const vk::Instance& VulkanContext::GetInstance()const {
	return instance_;
}

const std::vector<vk::PhysicalDevice>& VulkanContext::GetPhysicalDevices() const {
	return physicalDevices_;
}

Entity VulkanContext::CreateSurface(std::any& windowContext) {

	const Entity surface = entityManager_.CreateEntity();
	entityManager_.AttachComponent<VulkanSurface>(surface, this, windowContext);

	return surface;

}

std::unique_ptr<SwapChain> VulkanContext::CreateSwapChain(Entity device, Entity surface, glm::uvec2& size) {
	return std::make_unique<VulkanSwapChain>(entityManager_, device, surface, size);
}

Entity VulkanContext::CreateDevice(Entity surface) {

	const Entity device = entityManager_.CreateEntity();
	logicalDevices_.push_back(device);

	const auto& vkSurface = entityManager_.GetComponent<VulkanSurface>(surface).GetSurface();

	//TODO: proper device picking (based on a shared abstraction with Compute)

	int pickedDeviceIndex = -1;
	int pickedGraphicsIndex = -1;
	int pickedPresentIndex = -1;

	//iterate over all devices to investigate
	uint physicalDeviceIndex = 0;
	for (const auto& physicalDevice : physicalDevices_) {

		std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

		//reset index counts
		pickedGraphicsIndex = -1;
		pickedPresentIndex = -1;

		//iterate over all queue properties to find the right one
		uint queueFamilyIndex = 0;
		for (const auto& queueFamilyProperty : queueFamilyProperties) {

			//the properties we want
			if (queueFamilyProperty.queueCount > 0 && queueFamilyProperty.queueFlags & vk::QueueFlagBits::eGraphics) {
				pickedGraphicsIndex = queueFamilyIndex;
			}

			if (queueFamilyProperty.queueCount > 0 && physicalDevice.getSurfaceSupportKHR(queueFamilyIndex, vkSurface)) {
				pickedPresentIndex = queueFamilyIndex;
			}

			if (pickedGraphicsIndex >= 0 && pickedPresentIndex >= 0) {
				break;
			}

			++queueFamilyIndex;
		}

		//check extensions are fullfilled

		std::vector<vk::ExtensionProperties> availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();

		std::set<std::string> requiredExtensions(requiredDeviceExtensions_.begin(), requiredDeviceExtensions_.end());

		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		//swapchain supported
		bool swapChainAdequate = false;
		if (requiredExtensions.empty()) {

			vk::SurfaceCapabilitiesKHR capabilities = physicalDevice.getSurfaceCapabilitiesKHR(vkSurface);
			std::vector<vk::SurfaceFormatKHR> formats = physicalDevice.getSurfaceFormatsKHR(vkSurface);
			std::vector<vk::PresentModeKHR> presentModes = physicalDevice.getSurfacePresentModesKHR(vkSurface);

			swapChainAdequate = !formats.empty() && !presentModes.empty();
		}

		//conditions met to pick this physical device for the logical device
		if (pickedGraphicsIndex >= 0 && pickedPresentIndex >= 0 && requiredExtensions.empty() && swapChainAdequate) {
			pickedDeviceIndex = physicalDeviceIndex;
			break;
		}

		++physicalDeviceIndex;
	}

	//TODO: deal with families + device not found
	assert(pickedDeviceIndex >= 0);
	assert(pickedPresentIndex >= 0);
	assert(pickedGraphicsIndex >= 0);

	vk::PhysicalDevice& physicalDevice = physicalDevices_[pickedDeviceIndex];

	std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
	std::set<int> uniqueQueueFamilies = { pickedGraphicsIndex, pickedPresentIndex };

	//setup the device with the given index
	float queuePriority = 1.0f;

	for (int queueFamily : uniqueQueueFamilies) {

		vk::DeviceQueueCreateInfo deviceQueueCreateInfo;
		deviceQueueCreateInfo.flags = vk::DeviceQueueCreateFlags();
		deviceQueueCreateInfo.queueFamilyIndex = static_cast<uint>(queueFamily);
		deviceQueueCreateInfo.queueCount = 1;
		deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

		queueCreateInfos.push_back(deviceQueueCreateInfo);

	}

	vk::DeviceCreateInfo deviceCreateInfo;
	deviceCreateInfo.flags = vk::DeviceCreateFlags();
	deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
	deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
	deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtensions_.size());
	deviceCreateInfo.ppEnabledExtensionNames = requiredDeviceExtensions_.data();

	if (validationEnabled_) {
		deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		deviceCreateInfo.ppEnabledLayerNames = validationLayers.data();
	}
	else {
		deviceCreateInfo.enabledLayerCount = 0;
	}

	entityManager_.AttachComponent<VulkanDevice>(device, scheduler_, pickedGraphicsIndex, pickedPresentIndex,&physicalDevice, physicalDevice.createDevice(
		deviceCreateInfo
	));

	return device;
}

void VulkanContext::Synchronize() {

	for (auto& logicalDevice : logicalDevices_) {

		const auto& presentQueue = entityManager_.GetComponent<VulkanDevice>(logicalDevice).GetPresentQueue();

		presentQueue.waitIdle();

	}

}

VkBool32 VulkanContext::DebugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData) {

	//TODO: some true down to earth messaging
	return true;
}
