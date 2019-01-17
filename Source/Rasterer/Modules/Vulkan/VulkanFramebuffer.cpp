//#include "VulkanFramebuffer.h"
//
//#include "Composition/Entity/EntityManager.h"
//#include "VulkanDevice.h"
//#include "VulkanRenderPass.h"
//
//VulkanFramebuffer::VulkanFramebuffer(EntityManager& entityManager,Entity device, vk::ImageView& swapChainImageView, VulkanRenderPass& renderPass,  glm::uvec2& size) :
//	entityManager_(&entityManager),
//	device_(device)
//{
//
//	const auto& vkDevice = entityManager_->GetComponent<VulkanDevice>(device_);
//	const vk::Device& logicalDevice = vkDevice.GetLogicalDevice();
//
//	vk::ImageView attachments[] = {
//		swapChainImageView
//	};
//
//	vk::FramebufferCreateInfo framebufferInfo;
//	framebufferInfo.renderPass = renderPass.GetRenderPass();
//	framebufferInfo.attachmentCount = 1;
//	framebufferInfo.pAttachments = attachments;
//	framebufferInfo.width = size.x;
//	framebufferInfo.height = size.y;
//	framebufferInfo.layers = 1;
//
//
//	frameBuffer_ = logicalDevice.createFramebuffer(framebufferInfo);
//
//}
//
//VulkanFramebuffer::~VulkanFramebuffer() {
//
//	const auto& vkDevice = entityManager_->GetComponent<VulkanDevice>(device_);
//	const vk::Device& logicalDevice = vkDevice.GetLogicalDevice();
//
//	logicalDevice.destroyFramebuffer(frameBuffer_);
//
//}
//
//const vk::Framebuffer& VulkanFramebuffer::GetFrameBuffer() const {
//	return frameBuffer_;
//}
