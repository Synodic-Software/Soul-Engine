#pragma once

#include "Types.h"

#include <vulkan/vulkan.hpp>


class VulkanPhysicalDevice {

public:

	VulkanPhysicalDevice(vk::Instance, vk::PhysicalDevice);
	~VulkanPhysicalDevice();

	VulkanPhysicalDevice(const VulkanPhysicalDevice&) = default;
	VulkanPhysicalDevice(VulkanPhysicalDevice&&) noexcept = default;

	VulkanPhysicalDevice& operator=(const VulkanPhysicalDevice&) = default;
	VulkanPhysicalDevice& operator=(VulkanPhysicalDevice&&) noexcept = default;

	const vk::PhysicalDevice& Get();

private:

	vk::Instance instance_;
	vk::PhysicalDevice device_;

};
