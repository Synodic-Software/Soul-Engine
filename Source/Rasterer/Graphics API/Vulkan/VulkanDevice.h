#pragma once

#include "Rasterer/Graphics API/RasterDevice.h"
#include "Composition/Component/Component.h"
#include "Parallelism/Thread/ThreadLocal.h"

#include <vulkan/vulkan.hpp>

class Scheduler;

class VulkanDevice : public RasterDevice, Component<VulkanDevice> {

public:

	VulkanDevice(Scheduler&, int,int,vk::PhysicalDevice*, vk::Device);
	~VulkanDevice() override = default;

	VulkanDevice(const VulkanDevice&) = delete;
	VulkanDevice(VulkanDevice&& o) noexcept = default;

	VulkanDevice& operator=(const VulkanDevice&) = delete;
	VulkanDevice& operator=(VulkanDevice&& other) noexcept = default;

	//manually free resources as vulkan Instance terminates before entityManager
	void Terminate() override;

	void Rebuild();

	const vk::Device& GetLogicalDevice() const;
	const vk::PipelineCache& GetPipelineCache() const;
	const vk::PhysicalDevice& GetPhysicalDevice() const;
	const vk::CommandPool& GetCommandPool() const;
	const vk::Queue& GetGraphicsQueue() const;
	const vk::Queue& GetPresentQueue() const;

private:

	Scheduler* scheduler_;

	vk::Device device_;
	vk::PhysicalDevice* physicalDevice_;
	vk::PipelineCache pipelineCache_;

	vk::Queue graphicsQueue_;
	vk::Queue presentQueue_;

	ThreadLocal<vk::CommandPool> commandPool_;

	int graphicsIndex;
	int presentIndex;

	void Cleanup();
	void Generate();

};
