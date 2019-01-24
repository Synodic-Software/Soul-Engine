#include "VulkanDevice.h"

#include "Parallelism/Modules/Fiber/FiberParameters.h"
#include "Parallelism/Modules/Fiber/FiberScheduler.h"
#include "System/Compiler.h"

//TODO: Refactor
VulkanDevice::VulkanDevice(std::shared_ptr<FiberScheduler>& scheduler, vk::PhysicalDevice& physicalDevice) :
	scheduler_(scheduler),
	physicalDevice_(physicalDevice),
	graphicsIndex_(-1)
{

	//TODO: Swapchain not needed for certain embedded applications.
	const std::vector<const char*> requiredDeviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
		VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME
	};

	const std::vector<const char*> validationLayers = {
		"VK_LAYER_LUNARG_assistant_layer",
		"VK_LAYER_LUNARG_standard_validation"
	};


	deviceProperties_ = physicalDevice_.getProperties();
	deviceFeatures_ = physicalDevice_.getFeatures();
	memoryProperties_ = physicalDevice_.getMemoryProperties();

	std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
	std::vector<vk::ExtensionProperties> availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();

	std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

	//TODO: use priorities
	float queuePriority = 1.0f;

	// Used to add a queue 
	const auto addQueueInfo = [&queueCreateInfos, &queuePriority](uint32_t queueFamily)
	{

		vk::DeviceQueueCreateInfo deviceQueueCreateInfo;
		deviceQueueCreateInfo.flags = vk::DeviceQueueCreateFlags();
		deviceQueueCreateInfo.queueFamilyIndex = static_cast<uint>(queueFamily);

		//TODO: Allow multiple queues i.e 'queueFamilyProperties[queueFamily].queueCount'
		deviceQueueCreateInfo.queueCount = 1;
		deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

		queueCreateInfos.push_back(deviceQueueCreateInfo);

	};

	//Grab graphics queue
	for (uint i = 0; i < static_cast<uint>(queueFamilyProperties.size()); ++i)
	{
		if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics)
		{
			addQueueInfo(i);
			graphicsIndex_ = i;
			break;
		}
	}


	vk::DeviceCreateInfo deviceCreateInfo;
	deviceCreateInfo.flags = vk::DeviceCreateFlags();
	deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
	deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
	deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtensions.size());
	deviceCreateInfo.ppEnabledExtensionNames = requiredDeviceExtensions.data();

	if constexpr (Compiler::Debug()) {
		deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		deviceCreateInfo.ppEnabledLayerNames = validationLayers.data();
	}
	else {
		deviceCreateInfo.enabledLayerCount = 0;
	}

	logicalDevices_.push_back(physicalDevice.createDevice(deviceCreateInfo));

	//TODO: refactor queue 
	graphicsQueue_ = logicalDevices_[0].getQueue(graphicsIndex_, 0);

	//TODO: Refactor, should be an object instantiated by the device
	{
		vk::CommandPoolCreateInfo poolInfo;
		poolInfo.queueFamilyIndex = graphicsIndex_;

		//TODO: move to 3 commandpools per thread as suggested by NVIDIA
		scheduler_->ForEachThread(FiberPriority::UX, [this, poolInfo]()
		{

			//TODO: multiple logical devices
			commandPool_ = logicalDevices_[0].createCommandPool(poolInfo);

		});
	}

}

VulkanDevice::~VulkanDevice() {

	Synchronize();

	scheduler_->ForEachThread(FiberPriority::UX, [this]() noexcept
	{

		//TODO: multiple logical devices
		logicalDevices_[0].destroyCommandPool(commandPool_);

	});

	for (auto& logicalDevice : logicalDevices_)
	{

		logicalDevice.destroy();

	}

}

void VulkanDevice::Synchronize()
{

	for (auto& logicalDevice : logicalDevices_)
	{

		logicalDevice.waitIdle();

	}

}

const  vk::Device& VulkanDevice::GetLogical() const
{

	assert(!logicalDevices_.empty());
	//TODO: multiple logical devices
	return logicalDevices_[0];

}

const  vk::PhysicalDevice& VulkanDevice::GetPhysical() const
{

	return physicalDevice_;

}

const vk::CommandPool& VulkanDevice::GetCommandPool() const {

	return commandPool_;

}

//TODO: refactor queue 
const vk::Queue& VulkanDevice::GetGraphicsQueue() const {

	return graphicsQueue_;

}

//TODO: refactor queue 
const vk::Queue& VulkanDevice::GetPresentQueue() const {

	return graphicsQueue_;

}

//TODO: refactor queue 
int VulkanDevice::GetGraphicsIndex() const
{

	return graphicsIndex_;

}

SurfaceFormat VulkanDevice::GetSurfaceFormat(const vk::SurfaceKHR& surface) const
{

	const auto formats = physicalDevice_.getSurfaceFormatsKHR(surface);

	SurfaceFormat surfaceFormat = {
		vk::ColorSpaceKHR::eSrgbNonlinear,
		vk::Format::eB8G8R8A8Unorm,
	};

	//TODO: pick formats better
	if(!formats.empty() && formats.front().format == vk::Format::eUndefined)
	{
		return surfaceFormat;
	}
	
	for (const auto& format : formats) {

		if (format.format == vk::Format::eB8G8R8A8Unorm && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
			return surfaceFormat;
		}

	}

	surfaceFormat.colorFormat = formats.front().format;
	surfaceFormat.colorSpace = formats.front().colorSpace;

	return surfaceFormat;

}