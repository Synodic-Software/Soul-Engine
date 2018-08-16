#include "VulkanDevice.h"
#include "Parallelism/Fiber/Scheduler.h"

VulkanDevice::VulkanDevice(Scheduler& scheduler, vk::PhysicalDevice* physicalDevice, vk::Device device) :
	scheduler_(&scheduler),
	device_(device),
	physicalDevice_(physicalDevice)
{

	vk::PipelineCacheCreateInfo pipelineCreateInfo;
	//TODO: pipeline serialization n' such

	pipelineCache_ = device_.createPipelineCache(pipelineCreateInfo);

	//TODO: move to 3 commandpools per thread as suggest by NVIDIA
	scheduler_->ForEachThread(FiberPriority::UX,[this]()
	{

		vk::CommandPoolCreateInfo poolInfo;
		commandPool_ = device_.createCommandPool(poolInfo);

	});

	
}


void VulkanDevice::Terminate() {

	device_.destroyPipelineCache(pipelineCache_);

	scheduler_->ForEachThread(FiberPriority::UX, [this]()
	{

		device_.destroyCommandPool(commandPool_);

	});

	device_.destroy();

}

const vk::Device& VulkanDevice::GetLogicalDevice() const {
	return device_;
}

const vk::PipelineCache& VulkanDevice::GetPipelineCache() const {
	return pipelineCache_;
}

const vk::PhysicalDevice& VulkanDevice::GetPhysicalDevice() const {
	return *physicalDevice_;
}