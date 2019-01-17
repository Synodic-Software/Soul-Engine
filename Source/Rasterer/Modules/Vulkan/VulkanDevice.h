#pragma once

#include "Rasterer/RasterDevice.h"
#include "Parallelism/Thread/ThreadLocal.h"

#include <vulkan/vulkan.hpp>

class Scheduler;

class VulkanDevice final: public RasterDevice {

public:

	VulkanDevice(vk::PhysicalDevice&);
	~VulkanDevice() override;

	VulkanDevice(const VulkanDevice &) = delete;
	VulkanDevice(VulkanDevice &&) noexcept = default;

	VulkanDevice& operator=(const VulkanDevice &) = delete;
	VulkanDevice& operator=(VulkanDevice &&) noexcept = default;

	void Synchronize() override;

	void Rebuild();

private:

	std::vector<vk::Device> devices_;
	vk::PhysicalDevice physicalDevice_;
	//vk::PipelineCache pipelineCache_;

	//vk::Queue graphicsQueue_;
	//vk::Queue presentQueue_;

	//ThreadLocal<vk::CommandPool> commandPool_;

	//int graphicsIndex;
	//int presentIndex;

	void CreateQueues();
	void CreatePipelineCache();
	void Cleanup();


};
