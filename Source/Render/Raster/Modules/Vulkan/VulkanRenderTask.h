#pragma once

#include <vulkan/vulkan.hpp>

class VulkanRenderTask
{

public:

	VulkanRenderTask() = default;
	~VulkanRenderTask() = default;

	VulkanRenderTask(const VulkanRenderTask&) = delete;
	VulkanRenderTask(VulkanRenderTask&&) noexcept = delete;

	VulkanRenderTask& operator=(const VulkanRenderTask&) = delete;
	VulkanRenderTask& operator=(VulkanRenderTask&&) noexcept = delete;


private:


};
