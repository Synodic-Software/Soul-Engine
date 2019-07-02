#pragma once

#include "Types.h"

#include <vulkan/vulkan.hpp>

class VulkanSubPass {

public:

	VulkanSubPass(uint, std::vector<uint>);
	~VulkanSubPass() = default;

	VulkanSubPass(const VulkanSubPass&) = delete;
	VulkanSubPass(VulkanSubPass&&) noexcept = default;

	VulkanSubPass& operator=(const VulkanSubPass&) = delete;
	VulkanSubPass& operator=(VulkanSubPass&&) noexcept = default;

private:

	uint bindingIndex_;
	std::vector<uint> attachmentIndices_;

};


class VulkanRenderPass
{

public:

	VulkanRenderPass(const vk::Device&);
	~VulkanRenderPass();

	VulkanRenderPass(const VulkanRenderPass&) = delete;
	VulkanRenderPass(VulkanRenderPass&&) noexcept = default;

	VulkanRenderPass& operator=(const VulkanRenderPass&) = delete;
	VulkanRenderPass& operator=(VulkanRenderPass&&) noexcept = default;

	const vk::RenderPass& Get() const;

private:

	vk::Device device_;
	vk::RenderPass renderPass_;
	std::vector<vk::SubpassDescription> subpasses_;

};
