#pragma once

#include <vulkan/vulkan.hpp>

class VulkanDevice;

class VulkanRenderPass
{

public:

	VulkanRenderPass(std::shared_ptr<VulkanDevice>&, vk::Format);
	~VulkanRenderPass();

	VulkanRenderPass(const VulkanRenderPass&) = delete;
	VulkanRenderPass(VulkanRenderPass&&) noexcept = delete;

	VulkanRenderPass& operator=(const VulkanRenderPass&) = delete;
	VulkanRenderPass& operator=(VulkanRenderPass&&) noexcept = delete;

	const vk::RenderPass& GetRenderPass() const;

private:

	std::shared_ptr<VulkanDevice> device_;
	vk::RenderPass renderPass_;

};
