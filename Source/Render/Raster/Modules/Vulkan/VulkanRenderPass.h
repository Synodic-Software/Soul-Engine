#pragma once

#include <vulkan/vulkan.hpp>

class VulkanRenderPass
{

public:

	VulkanRenderPass(const vk::Device&, vk::Format);
	~VulkanRenderPass();

	VulkanRenderPass(const VulkanRenderPass&) = delete;
	VulkanRenderPass(VulkanRenderPass&&) noexcept = delete;

	VulkanRenderPass& operator=(const VulkanRenderPass&) = delete;
	VulkanRenderPass& operator=(VulkanRenderPass&&) noexcept = delete;

	const vk::RenderPass& Get() const;

private:

	vk::Device device_;
	vk::RenderPass renderPass_;
	std::vector<vk::SubpassDescription> subpasses_;

};
