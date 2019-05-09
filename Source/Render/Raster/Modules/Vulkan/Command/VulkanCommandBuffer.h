#pragma once

#include <memory>
#include <vulkan/vulkan.hpp>

class VulkanCommandPool;
class VulkanDevice;

class VulkanCommandBuffer final {

public:

	VulkanCommandBuffer(std::shared_ptr<VulkanCommandPool>&,
		const std::shared_ptr<VulkanDevice>&,
		vk::CommandBufferUsageFlagBits,
		vk::CommandBufferLevel);
	~VulkanCommandBuffer();

	VulkanCommandBuffer(const VulkanCommandBuffer&) = delete;
	VulkanCommandBuffer(VulkanCommandBuffer&&) noexcept = default;

	VulkanCommandBuffer& operator=(const VulkanCommandBuffer&) = delete;
	VulkanCommandBuffer& operator=(VulkanCommandBuffer&&) noexcept = default;

	void Begin();
	void End();

	void Submit();

	const vk::CommandBuffer& Get() const;


private:

	std::shared_ptr<VulkanCommandPool> commandPool_;
	std::shared_ptr<VulkanDevice> device_;


	vk::CommandBufferUsageFlagBits usage_;
	vk::CommandBuffer commandBuffer_;

};
