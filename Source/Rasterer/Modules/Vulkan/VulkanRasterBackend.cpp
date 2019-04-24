#include "VulkanRasterBackend.h"

#include "VulkanDevice.h"
#include "Core/Utility/Exception/Exception.h"
#include "Core/System/Compiler.h"
#include "Types.h"
#include "Display/Window/WindowModule.h"
#include "VulkanSwapChain.h"

VulkanRasterBackend::VulkanRasterBackend(std::shared_ptr<SchedulerModule>& scheduler,
	std::shared_ptr<WindowModule>& windowModule_):
	validationLayers_{
			"VK_LAYER_KHRONOS_validation"
}
{

	//setup Vulkan app info
	vk::ApplicationInfo appInfo;
	appInfo.apiVersion = VK_API_VERSION_1_1;
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);//TODO forward the application version here
	appInfo.pApplicationName = "Soul Engine"; //TODO forward the application name here
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0); //TODO forward the engine version here
	appInfo.pEngineName = "Soul Engine"; //TODO forward the engine name here

	// The display will forward the extensions needed for Vulkan
	const auto newExtensions = windowModule_->GetRasterExtensions();

	requiredInstanceExtensions_.insert(
		std::end(requiredInstanceExtensions_), std::begin(newExtensions), std::end(newExtensions));

	//TODO minimize memory/runtime impact
	if constexpr (Compiler::Debug()) {

		requiredInstanceExtensions_.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

		std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();

		for (auto layer : validationLayers_) {

			bool found = false;
			for (const auto& layerProperties : availableLayers) {

				if (strcmp(layer, layerProperties.layerName) == 0) {
					found = true;
					break;
				}

			}

			if (!found) {

				throw std::runtime_error("Specified Vulkan validation layer is not available.");

			}
		}

	}

	vk::InstanceCreateInfo instanceCreationInfo;
	instanceCreationInfo.pApplicationInfo = &appInfo;
	instanceCreationInfo.enabledExtensionCount = static_cast<uint32>(requiredInstanceExtensions_.size());
	instanceCreationInfo.ppEnabledExtensionNames = requiredInstanceExtensions_.data();

	if constexpr (Compiler::Debug()) {

		instanceCreationInfo.enabledLayerCount = static_cast<uint32>(validationLayers_.size());
		instanceCreationInfo.ppEnabledLayerNames = validationLayers_.data();

	}
	else {

		instanceCreationInfo.enabledLayerCount = 0;

	}

	instance_ = createInstance(instanceCreationInfo);


	//Create debugging callback
	if constexpr (Compiler::Debug()) {

		dispatcher_ = vk::DispatchLoaderDynamic(instance_);

		vk::DebugUtilsMessengerCreateInfoEXT messengerCreateInfo;
		messengerCreateInfo.flags = vk::DebugUtilsMessengerCreateFlagBitsEXT(0);
		messengerCreateInfo.messageSeverity =
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
		messengerCreateInfo.messageType =
			vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
			vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
			vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
		messengerCreateInfo.pfnUserCallback = DebugCallback;
		messengerCreateInfo.pUserData = nullptr;

		debugMessenger_ = instance_.createDebugUtilsMessengerEXT(messengerCreateInfo, nullptr, dispatcher_);

	}


	//setup devices
	auto physicalDevices = instance_.enumeratePhysicalDevices();
	devices_.reserve(physicalDevices.size());

	for (auto& physicalDevice : physicalDevices)
	{

		devices_.push_back(std::make_shared<VulkanDevice>(scheduler, physicalDevice));

	}

}

VulkanRasterBackend::~VulkanRasterBackend()
{

	for (auto& device : devices_) {

		device->Synchronize();

	}

	devices_.clear();

	if constexpr (Compiler::Debug()) {

		instance_.destroyDebugUtilsMessengerEXT(debugMessenger_, nullptr, dispatcher_);

	}

	instance_.destroy();

}

void VulkanRasterBackend::Draw()
{

	throw NotImplemented();

}

void VulkanRasterBackend::DrawIndirect()
{

	throw NotImplemented();

}

std::unique_ptr<VulkanSwapChain> VulkanRasterBackend::RegisterSurface(vk::SurfaceKHR& surface, glm::uvec2 size, VulkanSwapChain* oldSwapChain)
{

	//TODO: multiple devices
	auto& device = devices_[0];
	const vk::PhysicalDevice& physicalDevice = device->GetPhysical();

	if (!physicalDevice.getSurfaceSupportKHR(device->GetGraphicsIndex(), surface))
	{

		throw NotImplemented();

	}

	const auto format = device->GetSurfaceFormat(surface);
	return std::make_unique<VulkanSwapChain>(device, surface, format.colorFormat, format.colorSpace, size, false, oldSwapChain);

}

void VulkanRasterBackend::RemoveSurface(vk::SurfaceKHR& surface)
{

	instance_.destroySurfaceKHR(surface);

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

	throw NotImplemented();

}
