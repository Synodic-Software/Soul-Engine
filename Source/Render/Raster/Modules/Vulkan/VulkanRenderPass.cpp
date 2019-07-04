#include "VulkanRenderPass.h"
#include "Device/VulkanDevice.h"

VulkanSubPass::VulkanSubPass(uint bindingIndex, std::vector<uint> attachmentIndices):
	bindingIndex_(bindingIndex), 
	attachmentIndices_(std::move(attachmentIndices))
{
}

VulkanRenderPass::VulkanRenderPass(const vk::Device& device):
	device_(device)
{

	//vk::AttachmentDescription colorAttachment;
	//colorAttachment.format = swapChainImageFormat;
	//colorAttachment.samples = vk::SampleCountFlagBits::e1;
	//colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
	//colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
	//colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
	//colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
	//colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
	//colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

	//vk::AttachmentReference colorAttachmentRef;
	//colorAttachmentRef.attachment = 0;
	//colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

	//vk::SubpassDescription subpass;
	//subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
	//subpass.colorAttachmentCount = 1;
	//subpass.pColorAttachments = &colorAttachmentRef;

	vk::RenderPassCreateInfo renderPassInfo;
	renderPassInfo.attachmentCount = 0;
	renderPassInfo.pAttachments = nullptr;
	renderPassInfo.subpassCount = 0;
	renderPassInfo.pSubpasses = nullptr;

	renderPass_ = device_.createRenderPass(renderPassInfo, nullptr);

}

VulkanRenderPass::~VulkanRenderPass() {

	device_.destroyRenderPass(renderPass_);

}

const vk::RenderPass& VulkanRenderPass::Handle() const
{

	return renderPass_;

}
