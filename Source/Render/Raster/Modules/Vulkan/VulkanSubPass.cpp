#include "VulkanSubPass.h"

VulkanSubPass::VulkanSubPass(
	std::vector<vk::AttachmentReference2KHR> subpassAttachments)
{

	vk::SubpassDescription2KHR subPass;
	subPass.flags = vk::SubpassDescriptionFlags();
	subPass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
	subPass.viewMask = 0;
	subPass.inputAttachmentCount = 0;
	subPass.pInputAttachments = nullptr;
	subPass.colorAttachmentCount = subpassAttachments.size();
	subPass.pColorAttachments = subpassAttachments.data();
	subPass.pResolveAttachments = nullptr;
	subPass.pDepthStencilAttachment = nullptr;
	subPass.preserveAttachmentCount = 0;
	subPass.pPreserveAttachments = nullptr;

}