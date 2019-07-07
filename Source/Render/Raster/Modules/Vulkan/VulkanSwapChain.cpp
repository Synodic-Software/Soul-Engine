#include "VulkanSwapChain.h"

#include "Device/VulkanDevice.h"
#include "Core/Utility/Exception/Exception.h"

#include "Core/Geometry/Vertex.h"
#include "Buffer/VulkanBuffer.h"
#include "Transput/Resource/Resource.h"
#include "Render/Raster/Modules/Vulkan/VulkanSurface.h"


VulkanSwapChain::VulkanSwapChain(std::unique_ptr<VulkanDevice>& device,
	VulkanSurface& surface,
	bool vSync,
	VulkanSwapChain* oldSwapChain):
	device_(device->Logical()),
	currentFrame_(0), activeImageIndex_(0), frameMax_(2)
{
	const auto& physicalDevice = device->Physical();

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
	swapchainCreateInfo.clipped = VK_TRUE;
	swapchainCreateInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
	swapchainCreateInfo.oldSwapchain = oldSwapChain ? oldSwapChain->swapChain_ : nullptr;

	assert(device->SurfaceSupported(surface.Handle()));

	swapChain_ = device_.createSwapchainKHR(swapchainCreateInfo);
	auto swapChainImages = device_.getSwapchainImagesKHR(swapChain_);

	// set up synchronization primitives
	presentSemaphores_.resize(frameMax_);
	renderSemaphores_.resize(frameMax_);
	frameFences_.resize(frameMax_);

	vk::SemaphoreCreateInfo semaphoreInfo;

	vk::FenceCreateInfo fenceInfo;
	fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

	for (size_t i = 0; i < frameMax_; i++) {

		presentSemaphores_[i] = device_.createSemaphore(semaphoreInfo);
		renderSemaphores_[i] = device_.createSemaphore(semaphoreInfo);
		frameFences_[i] = device_.createFence(fenceInfo);
	}
}

VulkanSwapChain::~VulkanSwapChain()
{

	device_.waitIdle();

	for (size_t i = 0; i < frameMax_; i++) {

		device_.destroySemaphore(presentSemaphores_[i]);
		device_.destroySemaphore(renderSemaphores_[i]);
		device_.destroyFence(frameFences_[i]);
	}

	device_.destroySwapchainKHR(swapChain_);

}

void VulkanSwapChain::AquireImage(){

	auto [acquireResult, activeImageIndex_] = device_.acquireNextImageKHR(swapChain_,
		std::numeric_limits<uint64_t>::max(), presentSemaphores_[currentFrame_], nullptr);

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

vk::Extent2D VulkanSwapChain::GetSize()
{
	return size_;
}