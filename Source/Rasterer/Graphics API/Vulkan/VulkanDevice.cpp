#include "VulkanDevice.h"

VulkanDevice::VulkanDevice(vk::PhysicalDevice* physicalDevice, vk::Device device) :
	device_(device),
	physicalDevice_(physicalDevice)

{

	vk::PipelineCacheCreateInfo pipelineCreateInfo;
	//TODO: pipeline serialization n' such

	pipelineCache_ = device_.createPipelineCache(pipelineCreateInfo);

}


void VulkanDevice::Terminate() {

	device_.destroyPipelineCache(pipelineCache_);
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