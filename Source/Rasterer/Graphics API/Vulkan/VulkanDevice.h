#pragma once

#include "Rasterer/Graphics API/RasterDevice.h"
#include "Composition/Component/Component.h"

#include <vulkan/vulkan.hpp>

class VulkanDevice : public RasterDevice, Component<VulkanDevice> {

public:

	VulkanDevice(vk::PhysicalDevice*, vk::Device);
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

	vk::Device device_;
	vk::PhysicalDevice* physicalDevice_;
	vk::PipelineCache pipelineCache_;

};
