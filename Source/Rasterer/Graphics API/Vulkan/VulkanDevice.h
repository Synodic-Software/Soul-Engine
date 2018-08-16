#pragma once

#include "Rasterer/Graphics API/RasterDevice.h"
#include "Composition/Component/Component.h"
#include "Parallelism/Thread/ThreadLocal.h"

#include <vulkan/vulkan.hpp>

class Scheduler;

class VulkanDevice : public RasterDevice, Component<VulkanDevice> {

public:

	VulkanDevice(Scheduler&, vk::PhysicalDevice*, vk::Device);
	~VulkanDevice() override = default;

	VulkanDevice(const VulkanDevice&) = delete;
	VulkanDevice(VulkanDevice&& o) noexcept = default;

	VulkanDevice& operator=(const VulkanDevice&) = delete;
	VulkanDevice& operator=(VulkanDevice&& other) noexcept = default;

	//manually free resources as vulkan Instance terminates before entityManager
	void Terminate() override;

	const vk::Device& GetLogicalDevice() const;
	const vk::PipelineCache& GetPipelineCache() const;
	const vk::PhysicalDevice& GetPhysicalDevice() const;


private:

	Scheduler* scheduler_;

	vk::Device device_;
	vk::PhysicalDevice* physicalDevice_;
	vk::PipelineCache pipelineCache_;

	ThreadLocal<vk::CommandPool> commandPool_;

};
