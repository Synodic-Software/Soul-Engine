#include "VulkanRenderPass.h"
#include "VulkanDevice.h"

VulkanRenderPass::VulkanRenderPass(std::shared_ptr<VulkanDevice>& device, vk::Format swapChainImageFormat) :
	device_(device)
{

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

	const vk::Device& logicalDevice = device_->GetLogical();

	renderPass_ = logicalDevice.createRenderPass(renderPassInfo, nullptr);


}

VulkanRenderPass::~VulkanRenderPass() {

	const vk::Device& logicalDevice = device_->GetLogical();

	logicalDevice.destroyRenderPass(renderPass_);

}

const vk::RenderPass& VulkanRenderPass::GetRenderPass() const {

	return renderPass_;

}
