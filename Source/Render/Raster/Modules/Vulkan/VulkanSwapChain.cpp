#include "VulkanSwapChain.h"

#include "VulkanDevice.h"
#include "Core/Utility/Exception/Exception.h"

#include "Core/Geometry/Vertex.h"
#include "Buffer/VulkanBuffer.h"
#include "Transput/Resource/Resource.h"

VulkanSwapChain::VulkanSwapChain(std::shared_ptr<VulkanDevice>& device, vk::SurfaceKHR& surface,
	vk::Format colorFormat, vk::ColorSpaceKHR colorSpace, glm::uvec2& size, bool vSync, VulkanSwapChain* oldSwapChain) :
	vkDevice_(device),
	format_(colorFormat),
	size_(size), currentFrame_(0), frameMax_(2)
{
	const auto& logicalDevice = vkDevice_->GetLogical();
	const auto& physicalDevice = vkDevice_->GetPhysical();

	const vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
	std::vector<vk::PresentModeKHR> presentModes = physicalDevice.getSurfacePresentModesKHR(surface);

	assert(!presentModes.empty());

	vk::Extent2D swapchainSize;
	if (surfaceCapabilities.currentExtent.width == std::numeric_limits<uint32_t>::max())
	{
		swapchainSize.width = size_.x;
		swapchainSize.height = size_.y;
	}
	else
	{
		swapchainSize = surfaceCapabilities.currentExtent;
	}

	size_.x = swapchainSize.width;
	size_.y = swapchainSize.height;

	vk::PresentModeKHR swapchainPresentMode = vk::PresentModeKHR::eFifo;

	if (!vSync) {

		for (const auto& presentMode : presentModes) {
			if (presentMode == vk::PresentModeKHR::eMailbox) {
				swapchainPresentMode = vk::PresentModeKHR::eMailbox;
				break;
			}
			if (swapchainPresentMode != vk::PresentModeKHR::eMailbox && presentMode == vk::PresentModeKHR::eImmediate) {
				swapchainPresentMode = vk::PresentModeKHR::eImmediate;
			}
		}

	}
	else
	{
		throw NotImplemented();
	}

	vk::SurfaceTransformFlagBitsKHR preTransform;
	if (surfaceCapabilities.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity) {
		preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
	}
	else {
		preTransform = surfaceCapabilities.currentTransform;
	}

	//add one more image for buffering
	uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
	if (surfaceCapabilities.maxImageCount > 0 && imageCount > surfaceCapabilities.maxImageCount) {
		imageCount = surfaceCapabilities.maxImageCount;
	}

	vk::SwapchainCreateInfoKHR swapchainCreateInfo;
	swapchainCreateInfo.surface = surface;
	swapchainCreateInfo.minImageCount = imageCount;
	swapchainCreateInfo.imageFormat = format_;
	swapchainCreateInfo.imageColorSpace = colorSpace;
	swapchainCreateInfo.imageExtent = swapchainSize;
	swapchainCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;
	swapchainCreateInfo.preTransform = preTransform;
	swapchainCreateInfo.imageArrayLayers = 1;
	swapchainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
	swapchainCreateInfo.queueFamilyIndexCount = 0;
	swapchainCreateInfo.pQueueFamilyIndices = nullptr;
	swapchainCreateInfo.presentMode = swapchainPresentMode;
	swapchainCreateInfo.clipped = VK_TRUE;
	swapchainCreateInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
	swapchainCreateInfo.oldSwapchain = oldSwapChain ? oldSwapChain->swapChain_ : nullptr;

	swapChain_ = logicalDevice.createSwapchainKHR(swapchainCreateInfo);
	auto swapChainImages = logicalDevice.getSwapchainImagesKHR(swapChain_);

	vk::ImageViewCreateInfo colorAttachmentCreateInfo;
	colorAttachmentCreateInfo.format = format_;
	colorAttachmentCreateInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
	colorAttachmentCreateInfo.subresourceRange.levelCount = 1;
	colorAttachmentCreateInfo.subresourceRange.layerCount = 1;
	colorAttachmentCreateInfo.viewType = vk::ImageViewType::e2D;

	images_.resize(swapChainImages.size());
	for (uint32_t i = 0; i < swapChainImages.size(); ++i) {
		images_[i].image = swapChainImages[i];
		colorAttachmentCreateInfo.image = swapChainImages[i];
		images_[i].view = logicalDevice.createImageView(colorAttachmentCreateInfo);
		images_[i].fence = vk::Fence();
	}

	//set up synchronization primitives
	presentSemaphores_.resize(frameMax_);
	renderSemaphores_.resize(frameMax_);
	frameFences_.resize(frameMax_);

	vk::SemaphoreCreateInfo semaphoreInfo;

	vk::FenceCreateInfo fenceInfo;
	fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

	for (size_t i = 0; i < frameMax_; i++) {

		presentSemaphores_[i] = logicalDevice.createSemaphore(semaphoreInfo);
		renderSemaphores_[i] = logicalDevice.createSemaphore(semaphoreInfo);
		frameFences_[i] = logicalDevice.createFence(fenceInfo);

	}

}

VulkanSwapChain::~VulkanSwapChain() {

	const auto& logicalDevice = vkDevice_->GetLogical();

	vkDevice_->Synchronize();

	for (const auto& image : images_) {
		logicalDevice.destroyImageView(image.view);
	}

	for (size_t i = 0; i < frameMax_; i++) {

		logicalDevice.destroySemaphore(presentSemaphores_[i]);
		logicalDevice.destroySemaphore(renderSemaphores_[i]);
		logicalDevice.destroyFence(frameFences_[i]);

	}

	logicalDevice.destroySwapchainKHR(swapChain_);

}

void VulkanSwapChain::AquireImage()
{
	const auto& logicalDevice = vkDevice_->GetLogical();

	 auto [acquireResult, activeImageIndex_] = logicalDevice.acquireNextImageKHR(swapChain_,
		std::numeric_limits<uint64_t>::max(), presentSemaphores_[currentFrame_], nullptr);

	 if (acquireResult != vk::Result::eSuccess) {

	 	throw NotImplemented();

	 }

}


void VulkanSwapChain::Present(VulkanCommandBuffer& commandBuffer_)
{

	const auto& logicalDevice = vkDevice_->GetLogical();

	logicalDevice.waitForFences(frameFences_[currentFrame_], true, std::numeric_limits<uint64_t>::max());
	logicalDevice.resetFences(frameFences_[currentFrame_]);

	vk::SubmitInfo submitInfo;

	vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = &presentSemaphores_[currentFrame_];
	submitInfo.pWaitDstStageMask = waitStages;

	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer_.Get();

	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &renderSemaphores_[currentFrame_];

	vkDevice_->GetGraphicsQueue().submit(submitInfo, frameFences_[currentFrame_]);

	vk::PresentInfoKHR presentInfo;

	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &renderSemaphores_[currentFrame_];

	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &swapChain_;

	presentInfo.pImageIndices = &activeImageIndex_;

	vkDevice_->GetPresentQueue().presentKHR(presentInfo);

}

vk::Format VulkanSwapChain::GetFormat()
{
	return format_;
}