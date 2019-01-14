#include "VulkanRasterBackend.h"

#include "VulkanDevice.h"
#include "Core/Utility/Exception/Exception.h"
#include "System/Compiler.h"
#include "Core/Utility/Types.h"
#include <GLFW/glfw3.h>

VulkanRasterBackend::VulkanRasterBackend() :
	requiredDeviceExtensions_{ VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
		VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME
	},
	validationLayers{
			"VK_LAYER_LUNARG_assistant_layer",
			"VK_LAYER_LUNARG_standard_validation"
	}
{

	//setup Vulkan app info
	vk::ApplicationInfo applicationInfo;
	applicationInfo.apiVersion = VK_API_VERSION_1_1;
	applicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);//TODO forward the application version here
	applicationInfo.pApplicationName = "Soul Engine"; //TODO forward the application name here
	applicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0); //TODO forward the engine version here
	applicationInfo.pEngineName = "Soul Engine"; //TODO forward the engine name here

	//set instance extensions

	uint32 glfwExtensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	for (uint i = 0; i < glfwExtensionCount; ++i) {
		requiredInstanceExtensions_.push_back(glfwExtensions[i]);
	}

	//TODO minimize memory/runtime impact
	if constexpr (Compiler::Debug()) {

		requiredInstanceExtensions_.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

	}


	vk::InstanceCreateInfo instanceCreationInfo;
	instanceCreationInfo.pApplicationInfo = &applicationInfo;
	instanceCreationInfo.enabledExtensionCount = static_cast<uint32>(requiredInstanceExtensions_.size());
	instanceCreationInfo.ppEnabledExtensionNames = requiredInstanceExtensions_.data();

	if constexpr (Compiler::Debug()) {

		instanceCreationInfo.enabledLayerCount = static_cast<uint32>(validationLayers.size());
		instanceCreationInfo.ppEnabledLayerNames = validationLayers.data();

	}
	else {

		instanceCreationInfo.enabledLayerCount = 0;

	}

	instance_ = createInstance(instanceCreationInfo);

	//setup devices
	//TODO: abstract physical devices
	//TODO: guarantee size constnest
	physicalDevices_ = instance_.enumeratePhysicalDevices();

	//TODO should never trigger assert
	assert(!physicalDevices_.empty());

	//create debugging callback
	if constexpr (Compiler::Debug()) {

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

void VulkanRasterBackend::Draw()
{

	throw NotImplemented();

}

void VulkanRasterBackend::DrawIndirect()
{

	throw NotImplemented();

}

std::shared_ptr<RasterDevice> VulkanRasterBackend::CreateDevice()
{

	return std::shared_ptr<VulkanDevice>();

}

vk::Instance& VulkanRasterBackend::GetInstance()
{
	return instance_;
}

VkBool32 VulkanRasterBackend::DebugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData) {

	assert(false);
	//TODO: some true down to earth messaging
	return true;
}
