#pragma once

#include <vulkan/vulkan.hpp>
#include <glm/vec2.hpp>

class VulkanRenderPass;

class VulkanFrameBuffer{

public:

	VulkanFrameBuffer(const vk::Device& device,
		vk::ImageView& swapChainImageView,
		VulkanRenderPass&,
		glm::uvec2&);
	~VulkanFrameBuffer();

	VulkanFrameBuffer(const VulkanFrameBuffer&) = delete;
	VulkanFrameBuffer(VulkanFrameBuffer&& o) noexcept = default;

	VulkanFrameBuffer& operator=(const VulkanFrameBuffer&) = delete;
	VulkanFrameBuffer& operator=(VulkanFrameBuffer&& other) noexcept = default;


	const vk::Framebuffer& Handle() const;

private:

	vk::Device device_;
	vk::Framebuffer frameBuffer_;


};
