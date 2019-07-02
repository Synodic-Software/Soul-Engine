#include "VulkanDevice.h"

#include "Parallelism/Scheduler/TaskParameters.h"
#include "Parallelism/Scheduler/SchedulerModule.h"
#include "Core/System/Compiler.h"
#include "VulkanQueue.h"

VulkanDevice::VulkanDevice(std::shared_ptr<SchedulerModule>& scheduler,
	vk::PhysicalDevice& physicalDevice,
	nonstd::span<std::string> validationLayers):
	scheduler_(scheduler),
	physicalDevice_(physicalDevice)
{

	//Prepare input data
	// TODO: Expose extension configuration
	const std::vector<const char*> requiredDeviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME};

	std::vector<const char*> cValidationLayers(validationLayers.size());
	for (auto layer : validationLayers) {

		cValidationLayers.push_back(layer.c_str());
	}



	//
	deviceProperties_ = physicalDevice_.getProperties();
	deviceFeatures_ = physicalDevice_.getFeatures();
	memoryProperties_ = physicalDevice_.getMemoryProperties();

	std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
		physicalDevice.getQueueFamilyProperties();
	std::vector<vk::ExtensionProperties> availableExtensions =
		physicalDevice.enumerateDeviceExtensionProperties();

	std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

	// TODO: use priorities
	float queuePriority = 1.0f;

	// Used to add a queue
	const auto addQueueInfo = [&queueCreateInfos, &queuePriority](uint32_t queueFamily) {
		vk::DeviceQueueCreateInfo deviceQueueCreateInfo;
		deviceQueueCreateInfo.flags = vk::DeviceQueueCreateFlags();
		deviceQueueCreateInfo.queueFamilyIndex = static_cast<uint>(queueFamily);

		// TODO: Allow multiple queues i.e 'queueFamilyProperties[queueFamily].queueCount'
		deviceQueueCreateInfo.queueCount = 1;
		deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

		queueCreateInfos.push_back(deviceQueueCreateInfo);
	};

	// Iterate over all the queue families
	for (uint i = 0; i < static_cast<uint>(queueFamilyProperties.size()); ++i) {

		auto& queueFamily = queueFamilyProperties[i];

		if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
			addQueueInfo(i);
			break;
		}

	}

	vk::DeviceCreateInfo deviceCreateInfo;
	deviceCreateInfo.flags = vk::DeviceCreateFlags();
	deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
	deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
	deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtensions.size());
	deviceCreateInfo.ppEnabledExtensionNames = requiredDeviceExtensions.data();
	deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(cValidationLayers.size());
	deviceCreateInfo.ppEnabledLayerNames = cValidationLayers.data();

	auto& device = logicalDevices_.emplace_back(physicalDevice.createDevice(deviceCreateInfo));



}

VulkanDevice::~VulkanDevice()
{

	for (auto& logicalDevice : logicalDevices_) {

		logicalDevice.destroy();
	}
}

void VulkanDevice::Synchronize()
{

	for (auto& logicalDevice : logicalDevices_) {

		logicalDevice.waitIdle();
	}
}

const vk::Device& VulkanDevice::GetLogical() const
{

	assert(!logicalDevices_.empty());
	// TODO: multiple logical devices
	return logicalDevices_[0];
}

const vk::PhysicalDevice& VulkanDevice::GetPhysical() const
{

	return physicalDevice_;

}