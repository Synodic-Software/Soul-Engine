#pragma once

#include "Types.h"
#include "Device/VulkanDevice.h"
#include "VulkanSubPass.h"

#include <vulkan/vulkan.hpp>

class VulkanRenderPass
{

public:

	VulkanRenderPass(const VulkanDevice&,
		std::vector<vk::AttachmentDescription2KHR> subpassAttachments,
		std::vector<vk::SubpassDescription2KHR> subpassDescriptions,
		std::vector<vk::SubpassDependency2KHR> subpassDependencies);
	~VulkanRenderPass();

	VulkanRenderPass(const VulkanRenderPass&) = delete;
	VulkanRenderPass(VulkanRenderPass&&) noexcept = default;

	VulkanRenderPass& operator=(const VulkanRenderPass&) = delete;
	VulkanRenderPass& operator=(VulkanRenderPass&&) noexcept = default;

	const vk::RenderPass& Handle() const;

private:

	vk::Device device_;
	vk::RenderPass renderPass_;
	std::vector<VulkanSubPass> subpasses_;

};
