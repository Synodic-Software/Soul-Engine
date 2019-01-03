#include "VulkanRenderPass.h"

VulkanRenderPass::VulkanRenderPass(EntityManager& entityManager, Entity device, vk::Format swapChainImageFormat) :
	entityManager_(entityManager),
	device_(device)
{

	Create(swapChainImageFormat);

}

void VulkanRenderPass::Create(vk::Format swapChainImageFormat) {
	vk::AttachmentDescription colorAttachment;
	colorAttachment.format = swapChainImageFormat;
	colorAttachment.samples = vk::SampleCountFlagBits::e1;
	colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
	colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
	colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
	colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
	colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
	colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

	vk::AttachmentReference colorAttachmentRef;
	colorAttachmentRef.attachment = 0;
	colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

	vk::SubpassDescription subpass;
	subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;

	vk::RenderPassCreateInfo renderPassInfo;
	renderPassInfo.attachmentCount = 1;
	renderPassInfo.pAttachments = &colorAttachment;
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;

	const auto& vkDevice = entityManager_.GetComponent<VulkanDevice>(device_);
	const vk::Device& logicalDevice = vkDevice.GetLogicalDevice();

	renderPass_ = logicalDevice.createRenderPass(renderPassInfo, nullptr);

}

VulkanRenderPass::~VulkanRenderPass() {

	const auto& vkDevice = entityManager_.GetComponent<VulkanDevice>(device_);
	const vk::Device& logicalDevice = vkDevice.GetLogicalDevice();

	logicalDevice.destroyRenderPass(renderPass_, nullptr);

}

const vk::RenderPass& VulkanRenderPass::GetRenderPass() const {
	return renderPass_;
}
