#include "VulkanRenderPass.h"

VulkanSubPass::VulkanSubPass(uint bindingIndex, std::vector<uint> attachmentIndices):
	bindingIndex_(bindingIndex), 
	attachmentIndices_(std::move(attachmentIndices))
{
}

VulkanRenderPass::VulkanRenderPass(const VulkanDevice& device):
	device_(device.Logical())
{

	//vk::AttachmentDescription2KHR colorAttachment;
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

	std::vector<vk::SubpassDescription2KHR> subpassDescriptions(subpasses_.size());
	std::vector<vk::SubpassDependency> subpassDependencies(subpasses_.size());

	vk::RenderPassCreateInfo2KHR renderPassInfo;
	renderPassInfo.flags = vk::RenderPassCreateFlags();
	renderPassInfo.attachmentCount = 0;
	renderPassInfo.pAttachments = nullptr;
	renderPassInfo.subpassCount = subpassDescriptions.size();
	renderPassInfo.pSubpasses = subpassDescriptions.data();
	renderPassInfo.dependencyCount = 0;
	renderPassInfo.pDependencies = nullptr;
	renderPassInfo.correlatedViewMaskCount = 0;
	renderPassInfo.pCorrelatedViewMasks = nullptr;

	renderPass_ = device_.createRenderPass2KHR(renderPassInfo, nullptr, device.DispatchLoader());

}

VulkanRenderPass::~VulkanRenderPass() {

	device_.destroyRenderPass(renderPass_);

}

const vk::RenderPass& VulkanRenderPass::Handle() const
{

	return renderPass_;

}
