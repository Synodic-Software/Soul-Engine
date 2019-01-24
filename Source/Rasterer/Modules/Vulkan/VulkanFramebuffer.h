#pragma once

#include <vulkan/vulkan.hpp>
#include <glm/vec2.hpp>

class VulkanRenderPass;
class EntityManager;
class VulkanDevice;

class VulkanFrameBuffer{

public:

	VulkanFrameBuffer(std::shared_ptr<VulkanDevice>&, vk::ImageView& swapChainImageView, VulkanRenderPass&, glm::uvec2&);
	~VulkanFrameBuffer();

	VulkanFrameBuffer(const VulkanFrameBuffer&) = delete;
	VulkanFrameBuffer(VulkanFrameBuffer&& o) noexcept = default;

	VulkanFrameBuffer& operator=(const VulkanFrameBuffer&) = delete;
	VulkanFrameBuffer& operator=(VulkanFrameBuffer&& other) noexcept = default;


	const vk::Framebuffer& GetFrameBuffer() const;

private:
	
	vk::Framebuffer frameBuffer_;

	std::shared_ptr<VulkanDevice> device_;

};
