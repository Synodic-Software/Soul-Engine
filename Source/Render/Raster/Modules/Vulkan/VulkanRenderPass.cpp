#include "VulkanRenderPass.h"

VulkanRenderPass::VulkanRenderPass(const VulkanDevice& device,
	std::vector<vk::AttachmentDescription2KHR> subpassAttachments,
	std::vector<vk::SubpassDescription2KHR> subpassDescriptions,
	std::vector<vk::SubpassDependency2KHR> subpassDependencies):
	device_(device.Logical())
{

	vk::RenderPassCreateInfo2KHR renderPassInfo;
	renderPassInfo.flags = vk::RenderPassCreateFlags();
	renderPassInfo.attachmentCount = static_cast<uint>(subpassAttachments.size());
	renderPassInfo.pAttachments = subpassAttachments.data();
	renderPassInfo.subpassCount = static_cast<uint>(subpassDescriptions.size());
	renderPassInfo.pSubpasses = subpassDescriptions.data();
	renderPassInfo.dependencyCount = static_cast<uint>(subpassDependencies.size());
	renderPassInfo.pDependencies = subpassDependencies.data();
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
