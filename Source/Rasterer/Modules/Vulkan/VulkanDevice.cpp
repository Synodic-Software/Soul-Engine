#include "VulkanDevice.h"

VulkanDevice::VulkanDevice(vk::PhysicalDevice& physicalDevice) :
	physicalDevice_(physicalDevice)

{

	//CreateQueues();

//CreatePipelineCache();

//vk::CommandPoolCreateInfo poolInfo;
//poolInfo.queueFamilyIndex = graphicsIndex;

////TODO: move to 3 commandpools per thread as suggested by NVIDIA
//scheduler_->ForEachThread(FiberPriority::UX, [this, poolInfo]()
//{

//	commandPool_ = device_.createCommandPool(poolInfo);

//});


}

VulkanDevice::~VulkanDevice() {

	/*Cleanup();

	scheduler_->ForEachThread(FiberPriority::UX, [this]()
	{

		device_.destroyCommandPool(commandPool_);

	});

	device_.destroy();*/

}

void VulkanDevice::Synchronize()
{

}

void VulkanDevice::CreateQueues() {

	//graphicsQueue_ = device_.getQueue(graphicsIndex, 0);
	//presentQueue_ = device_.getQueue(presentIndex, 0);

}

void VulkanDevice::CreatePipelineCache() {

	//vk::PipelineCacheCreateInfo pipelineCreateInfo;
	////TODO: pipeline serialization n' such

	//pipelineCache_ = device_.createPipelineCache(pipelineCreateInfo);

}

void VulkanDevice::Cleanup() {

	//device_.destroyPipelineCache(pipelineCache_);

}

void VulkanDevice::Rebuild() {

	//Cleanup();

	////	CreateQueues();
	//CreatePipelineCache();

}