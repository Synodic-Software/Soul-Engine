#pragma once

#include "Rasterer/RasterDevice.h"
#include "Parallelism/Thread/ThreadLocal.h"

#include <vulkan/vulkan.hpp>

class Scheduler;

class VulkanDevice final: public RasterDevice {

public:

	VulkanDevice(Scheduler&, int, int, vk::PhysicalDevice*, vk::Device);
	~VulkanDevice() override;

	VulkanDevice(const VulkanDevice &) = delete;
	VulkanDevice(VulkanDevice &&) noexcept = default;

	VulkanDevice& operator=(const VulkanDevice &) = delete;
	VulkanDevice& operator=(VulkanDevice &&) noexcept = default;

	void Rebuild();

	const vk::Device& GetLogicalDevice() const;
	const vk::PipelineCache& GetPipelineCache() const;
	const vk::PhysicalDevice& GetPhysicalDevice() const;
	const vk::CommandPool& GetCommandPool() const;
	const vk::Queue& GetGraphicsQueue() const;
	const vk::Queue& GetPresentQueue() const;

private:

	vk::Device device_;
	vk::PhysicalDevice* physicalDevice_;
	vk::PipelineCache pipelineCache_;

	vk::Queue graphicsQueue_;
	vk::Queue presentQueue_;

	ThreadLocal<vk::CommandPool> commandPool_;

	int graphicsIndex;
	int presentIndex;

	void CreateQueues();
	void CreatePipelineCache();
	void Cleanup();
	void Generate();

	Scheduler* scheduler_;

};
