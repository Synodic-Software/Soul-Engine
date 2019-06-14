#pragma once

#include "Types.h"

#include <vulkan/vulkan.hpp>


class VulkanQueue {

public:

	VulkanQueue(const vk::Device& device, uint index, uint familyIndex);
	~VulkanQueue() = default;

	VulkanQueue(const VulkanQueue&) = default;
	VulkanQueue(VulkanQueue&&) noexcept = default;

	VulkanQueue& operator=(const VulkanQueue&) = default;
	VulkanQueue& operator=(VulkanQueue&&) noexcept = default;

	bool Submit();
	bool Present();

	const vk::Queue& Get();

private:

	vk::Device device_;
	vk::Queue queue_;

	const uint index_;
	const uint familyIndex_;


};
