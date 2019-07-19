#pragma once

#include "Types.h"
#include "Device/VulkanDevice.h"

#include <vulkan/vulkan.hpp>

class VulkanSubPass {

public:

	VulkanSubPass(std::vector<vk::AttachmentReference2KHR> outputAttachmentReferences);
	~VulkanSubPass() = default;

	VulkanSubPass(const VulkanSubPass&) = delete;
	VulkanSubPass(VulkanSubPass&&) noexcept = default;

	VulkanSubPass& operator=(const VulkanSubPass&) = delete;
	VulkanSubPass& operator=(VulkanSubPass&&) noexcept = default;


};
