#include "VulkanFramebuffer.h"

#include "VulkanRenderPass.h"

VulkanFrameBuffer::VulkanFrameBuffer(const vk::Device& device,
	const std::vector<vk::ImageView>& attachments,
	VulkanRenderPass& renderPass,
	glm::uvec2& size):
	device_(device)
{

	vk::FramebufferCreateInfo framebufferInfo;
	framebufferInfo.flags = vk::FramebufferCreateFlags();
	framebufferInfo.renderPass = renderPass.Handle();
	framebufferInfo.attachmentCount = attachments.size();
	framebufferInfo.pAttachments = attachments.data();
	framebufferInfo.width = size.x;
	framebufferInfo.height = size.y;
	framebufferInfo.layers = 1;


	frameBuffer_ = device_.createFramebuffer(framebufferInfo);

}

VulkanFrameBuffer::~VulkanFrameBuffer() {

	device_.destroyFramebuffer(frameBuffer_);

}

const vk::Framebuffer& VulkanFrameBuffer::Handle() const
{

	return frameBuffer_;

}
