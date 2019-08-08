#include "VulkanSwapChain.h"

#include "Device/VulkanDevice.h"
#include "Core/Utility/Exception/Exception.h"

#include "Core/Geometry/Vertex.h"
#include "Buffer/VulkanBuffer.h"
#include "Transput/Resource/Resource.h"
#include "Render/Raster/Modules/Vulkan/VulkanSurface.h"


VulkanSwapChain::VulkanSwapChain(VulkanDevice& device,
	VulkanSurface& surface,
	bool vSync,
	VulkanSwapChain* oldSwapChain):
	device_(device.Logical()),
	activeImageIndex_(0)
{
	auto& logicalDevice = device.Logical();
	const auto& physicalDevice = device.Physical();

	const vk::SurfaceCapabilitiesKHR surfaceCapabilities =
		physicalDevice.getSurfaceCapabilitiesKHR(surface.Handle());
	std::vector<vk::PresentModeKHR> presentModes =
		physicalDevice.getSurfacePresentModesKHR(surface.Handle());

	assert(!presentModes.empty());

	vk::Extent2D swapchainSize;
	if (surfaceCapabilities.currentExtent.width == std::numeric_limits<uint32_t>::max()) {
		swapchainSize = size_;
	}
	else {
		swapchainSize = surfaceCapabilities.currentExtent;
	}

	size_ = swapchainSize;

	vk::PresentModeKHR swapchainPresentMode = vk::PresentModeKHR::eFifo;

	if (!vSync) {

		for (const auto& presentMode : presentModes) {
			if (presentMode == vk::PresentModeKHR::eMailbox) {
				swapchainPresentMode = vk::PresentModeKHR::eMailbox;
				break;
			}
			if (swapchainPresentMode != vk::PresentModeKHR::eMailbox &&
				presentMode == vk::PresentModeKHR::eImmediate) {
				swapchainPresentMode = vk::PresentModeKHR::eImmediate;
			}
		}
	}
	else {
		throw NotImplemented();
	}

	vk::SurfaceTransformFlagBitsKHR preTransform;
	if (surfaceCapabilities.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity) {
		preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
	}
	else {
		preTransform = surfaceCapabilities.currentTransform;
	}

	// add one more image for buffering
	uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
	if (surfaceCapabilities.maxImageCount > 0 && imageCount > surfaceCapabilities.maxImageCount) {
		imageCount = surfaceCapabilities.maxImageCount;
	}

	vk::SurfaceFormatKHR format = surface.Format();

	vk::SwapchainCreateInfoKHR swapchainCreateInfo;
	swapchainCreateInfo.surface = surface.Handle();
	swapchainCreateInfo.minImageCount = imageCount;
	swapchainCreateInfo.imageFormat = format.format;
	swapchainCreateInfo.imageColorSpace = format.colorSpace;
	swapchainCreateInfo.imageExtent = swapchainSize;
	swapchainCreateInfo.imageUsage =
		vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;
	swapchainCreateInfo.preTransform = preTransform;
	swapchainCreateInfo.imageArrayLayers = 1;
	swapchainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
	swapchainCreateInfo.queueFamilyIndexCount = 0;
	swapchainCreateInfo.pQueueFamilyIndices = nullptr;
	swapchainCreateInfo.presentMode = swapchainPresentMode;
	swapchainCreateInfo.clipped = true;
	swapchainCreateInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
	swapchainCreateInfo.oldSwapchain = oldSwapChain ? oldSwapChain->swapChain_ : nullptr;

	assert(device.SurfaceSupported(surface.Handle()));

	swapChain_ = device_.createSwapchainKHR(swapchainCreateInfo);
	renderImages_ = device_.getSwapchainImagesKHR(swapChain_);
	renderImageViews_.resize(renderImages_.size());

	for (auto i = 0; i < renderImages_.size(); ++i) {

		vk::ImageViewCreateInfo imageViewCreateInfo;
		imageViewCreateInfo.flags = vk::ImageViewCreateFlags();
		imageViewCreateInfo.image = renderImages_[i];
		imageViewCreateInfo.viewType = vk::ImageViewType::e2D;
		imageViewCreateInfo.format = format.format;
		imageViewCreateInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
		imageViewCreateInfo.subresourceRange.levelCount = 1;
		imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
		imageViewCreateInfo.subresourceRange.layerCount = 1;

		renderImageViews_[i] = logicalDevice.createImageView(imageViewCreateInfo);

	}

}

VulkanSwapChain::~VulkanSwapChain()
{

	for (const auto& imageView : renderImageViews_) {
		device_.destroyImageView(imageView);
	}

	device_.waitIdle();
	device_.destroySwapchainKHR(swapChain_);

}

nonstd::span<vk::Image> VulkanSwapChain::Images()
{

	return {renderImages_};

}

nonstd::span<vk::ImageView> VulkanSwapChain::ImageViews()
{

	return {renderImageViews_};

}

uint VulkanSwapChain::ActiveImageIndex() const
{

	return activeImageIndex_;

}

void VulkanSwapChain::AquireImage(const vk::Semaphore& presentSemaphore)
{

	auto [acquireResult, activeImageIndex_] = device_.acquireNextImageKHR(swapChain_, std::numeric_limits<uint64_t>::max(), presentSemaphore, nullptr);

	if (acquireResult != vk::Result::eSuccess) {

		throw NotImplemented();

	}

}


//void VulkanSwapChain::Present(VulkanCommandBuffer& commandBuffer_)
//{
//
//	const auto& logicalDevice = vkDevice_->GetLogical();
//
//	logicalDevice.waitForFences(
//		frameFences_[currentFrame_], true, std::numeric_limits<uint64_t>::max());
//	logicalDevice.resetFences(frameFences_[currentFrame_]);
//
//	vk::SubmitInfo submitInfo;
//
//	vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
//	submitInfo.waitSemaphoreCount = 1;
//	submitInfo.pWaitSemaphores = &presentSemaphores_[currentFrame_];
//	submitInfo.pWaitDstStageMask = waitStages;
//
//	submitInfo.commandBufferCount = 1;
//	submitInfo.pCommandBuffers = &commandBuffer_.Handle();
//
//	submitInfo.signalSemaphoreCount = 1;
//	submitInfo.pSignalSemaphores = &renderSemaphores_[currentFrame_];
//
//	vkDevice_->GetGraphicsQueue().submit(submitInfo, frameFences_[currentFrame_]);
//
//	vk::PresentInfoKHR presentInfo;
//
//	presentInfo.waitSemaphoreCount = 1;
//	presentInfo.pWaitSemaphores = &renderSemaphores_[currentFrame_];
//
//	presentInfo.swapchainCount = 1;
//	presentInfo.pSwapchains = &swapChain_;
//
//	presentInfo.pImageIndices = &activeImageIndex_;
//
//	vkDevice_->GetPresentQueue().presentKHR(presentInfo);
//}

vk::Extent2D VulkanSwapChain::Size()
{
	return size_;
}