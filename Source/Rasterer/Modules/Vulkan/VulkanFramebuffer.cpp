#include "VulkanFrameBuffer.h"

#include "Composition/Entity/EntityManager.h"
#include "VulkanDevice.h"
#include "VulkanRenderPass.h"

VulkanFrameBuffer::VulkanFrameBuffer(std::shared_ptr<VulkanDevice>& device, vk::ImageView& swapChainImageView, VulkanRenderPass& renderPass,  glm::uvec2& size) :
	device_(device)
{

	const vk::Device& logicalDevice = device_->GetLogical();

	vk::ImageView attachments[] = {
		swapChainImageView
	};

	vk::FramebufferCreateInfo framebufferInfo;
	framebufferInfo.renderPass = renderPass.GetRenderPass();
	framebufferInfo.attachmentCount = 1;
	framebufferInfo.pAttachments = attachments;
	framebufferInfo.width = size.x;
	framebufferInfo.height = size.y;
	framebufferInfo.layers = 1;


	frameBuffer_ = logicalDevice.createFramebuffer(framebufferInfo);

}

VulkanFrameBuffer::~VulkanFrameBuffer() {

	const vk::Device& logicalDevice = device_->GetLogical();

	logicalDevice.destroyFramebuffer(frameBuffer_);

}

const vk::Framebuffer& VulkanFrameBuffer::GetFrameBuffer() const {
	return frameBuffer_;
}
