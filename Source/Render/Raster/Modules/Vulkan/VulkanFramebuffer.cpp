#include "VulkanFramebuffer.h"

#include "VulkanRenderPass.h"

VulkanFrameBuffer::VulkanFrameBuffer(const vk::Device& device,
	vk::ImageView& swapChainImageView,
	VulkanRenderPass& renderPass,
	glm::uvec2& size):
	device_(device)
{

	vk::ImageView attachments[] = {
		swapChainImageView
	};

	vk::FramebufferCreateInfo framebufferInfo;
	framebufferInfo.renderPass = renderPass.Get();
	framebufferInfo.attachmentCount = 1;
	framebufferInfo.pAttachments = attachments;
	framebufferInfo.width = size.x;
	framebufferInfo.height = size.y;
	framebufferInfo.layers = 1;


	frameBuffer_ = device_.createFramebuffer(framebufferInfo);

}

VulkanFrameBuffer::~VulkanFrameBuffer() {

	device_.destroyFramebuffer(frameBuffer_);

}

const vk::Framebuffer& VulkanFrameBuffer::Get() const {

	return frameBuffer_;

}
