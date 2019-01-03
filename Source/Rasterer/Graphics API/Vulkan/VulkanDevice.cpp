#include "VulkanDevice.h"
#include "Parallelism/Fiber/Scheduler.h"

VulkanDevice::VulkanDevice(Scheduler& scheduler, int gIndex, int pIndex, vk::PhysicalDevice* physicalDevice, vk::Device device) :
	scheduler_(&scheduler),
	device_(device),
	physicalDevice_(physicalDevice),
	graphicsIndex(gIndex),
	presentIndex(pIndex)
{

	Generate();

}

VulkanDevice::~VulkanDevice() {

	Cleanup();

	scheduler_->ForEachThread(FiberPriority::UX, [this]()
	{

		device_.destroyCommandPool(commandPool_);

	});

	device_.destroy();

}

void VulkanDevice::Generate() {

	CreateQueues();

	CreatePipelineCache();

	vk::CommandPoolCreateInfo poolInfo;
	poolInfo.queueFamilyIndex = graphicsIndex;

	//TODO: move to 3 commandpools per thread as suggested by NVIDIA
	scheduler_->ForEachThread(FiberPriority::UX, [this, poolInfo]()
	{

		commandPool_ = device_.createCommandPool(poolInfo);

	});

}

void VulkanDevice::CreateQueues() {

	graphicsQueue_ = device_.getQueue(graphicsIndex,0);
	presentQueue_ = device_.getQueue(presentIndex,0);

}

void VulkanDevice::CreatePipelineCache() {

	vk::PipelineCacheCreateInfo pipelineCreateInfo;
	//TODO: pipeline serialization n' such

	pipelineCache_ = device_.createPipelineCache(pipelineCreateInfo);

}

void VulkanDevice::Cleanup() {

	device_.destroyPipelineCache(pipelineCache_);

}

void VulkanDevice::Rebuild() {

	Cleanup();

//	CreateQueues();
	CreatePipelineCache();

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

const vk::CommandPool& VulkanDevice::GetCommandPool() const {
	return commandPool_;
}

const vk::Queue& VulkanDevice::GetGraphicsQueue() const {
	return graphicsQueue_;
}

const vk::Queue& VulkanDevice::GetPresentQueue() const {
	return presentQueue_;
}
