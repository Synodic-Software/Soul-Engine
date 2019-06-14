#pragma once

#include "Types.h"

#include <vulkan/vulkan.hpp>


class VulkanInstance {

public:

	VulkanInstance();
	~VulkanInstance();

	VulkanInstance(const VulkanInstance&) = default;
	VulkanInstance(VulkanInstance&&) noexcept = default;

	VulkanInstance& operator=(const VulkanInstance&) = default;
	VulkanInstance& operator=(VulkanInstance&&) noexcept = default;

	const vk::Instance& Get();

private:

	vk::Instance instance_;

};
