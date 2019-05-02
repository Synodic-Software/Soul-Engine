#include "VulkanSwapChain.h"

#include "VulkanDevice.h"
#include "Core/Utility/Exception/Exception.h"

#include "Core/Geometry/Vertex.h"
#include "Buffer/VulkanBuffer.h"
#include "Transput/Resource/Resource.h"

VulkanSwapChain::VulkanSwapChain(std::shared_ptr<VulkanDevice>& device, vk::SurfaceKHR& surface,
	vk::Format colorFormat, vk::ColorSpaceKHR colorSpace, glm::uvec2& size, bool vSync, VulkanSwapChain* oldSwapChain) :
	vkDevice_(device),
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
	swapchainCreateInfo.imageFormat = colorFormat;
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
	colorAttachmentCreateInfo.format = colorFormat;
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


	//TODO: Remove hardcoded pipeline + Hardcoded paths
	//TODO: Associate paths to Project/Executable
	//pipeline_ = std::make_unique<VulkanPipeline>(vkDevice_, swapchainSize, Resource("../Resources/Shaders/vert.spv"), Resource("../Resources/Shaders/frag.spv"), colorFormat);

	/*frameBuffers_.reserve(images_.size());
	for (SwapChainImage& image : images_) {
		frameBuffers_.emplace_back(vkDevice_, image.view, pipeline_->GetRenderPass(), size);
	}*/

	/*vk::CommandBufferAllocateInfo allocInfo;
	allocInfo.commandPool = vkDevice_->GetCommandPool();
	allocInfo.level = vk::CommandBufferLevel::ePrimary;
	allocInfo.commandBufferCount = static_cast<uint32_t>(swapChainImages.size());

	commandBuffers_

	commandBuffers_ = logicalDevice.allocateCommandBuffers(allocInfo);

	vk::ClearValue clearColor(
		vk::ClearColorValue(std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f })
	);

	for (size_t i = 0; i < commandBuffers_.size(); i++) {

		vk::CommandBufferBeginInfo beginInfo;
		beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
		beginInfo.pInheritanceInfo = nullptr;

		vk::RenderPassBeginInfo renderPassInfo;
		renderPassInfo.renderPass = pipeline_->GetRenderPass().GetRenderPass();
		renderPassInfo.framebuffer = frameBuffers_[i].GetFrameBuffer();
		renderPassInfo.renderArea.offset = vk::Offset2D(0, 0);
		renderPassInfo.renderArea.extent = swapchainSize;
		renderPassInfo.clearValueCount = 1;
		renderPassInfo.pClearValues = &clearColor;

		commandBuffers_[i].begin(beginInfo);

		commandBuffers_[i].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
		commandBuffers_[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline_->GetPipeline());

        vk::Buffer vertexBuffers[] = { pipeline_->GetVertexBuffer().GetBuffer() };
		vk::DeviceSize offsets[] = { 0 };

		commandBuffers_[i].bindVertexBuffers(0, 1, vertexBuffers, offsets);
		commandBuffers_[i].bindIndexBuffer(pipeline_->GetIndexBuffer().GetBuffer(), 0, vk::IndexType::eUint16);

		commandBuffers_[i].drawIndexed(6, 1, 0, 0, 0);
		commandBuffers_[i].endRenderPass();

		commandBuffers_[i].end();

	}*/


	////set up synchronization primitives
	//imageAvailableSemaphores.resize(flightFramesCount);
	//renderFinishedSemaphores.resize(flightFramesCount);
	//fences_.resize(frameMax_);

	////vk::SemaphoreCreateInfo semaphoreInfo;

	//vk::FenceCreateInfo fenceInfo;
	//fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

	//for (size_t i = 0; i < frameMax_; i++) {

	//	fences_[i] = logicalDevice.createFence(fenceInfo);

	//}
}

VulkanSwapChain::~VulkanSwapChain() {

	const auto& logicalDevice = vkDevice_->GetLogical();

	//vkDevice_->Synchronize();

	//frameBuffers_.clear();

	for (const auto& image : images_) {
		logicalDevice.destroyImageView(image.view);
	}

	//for (auto& fence : fences_) {

	//	//logicalDevice.destroySemaphore(imageAvailableSemaphores[i]);
	//	//logicalDevice.destroySemaphore(renderFinishedSemaphores[i]);
	//	logicalDevice.destroyFence(fence);

	//}

	logicalDevice.destroySwapchainKHR(swapChain_);

}

void VulkanSwapChain::AquireImage(const vk::Semaphore& presentSemaphore)
{
	const auto& logicalDevice = vkDevice_->GetLogical();

	 auto[acquireResult, activeImageIndex_] = logicalDevice.acquireNextImageKHR(swapChain_, std::numeric_limits<uint64_t>::max(), presentSemaphore, nullptr);

	 if (acquireResult != vk::Result::eSuccess) {

	 	throw NotImplemented();

	 }

}


void VulkanSwapChain::Present(const vk::Queue& presentQueue, const vk::Semaphore& waitSemaphore)
{

	const auto& logicalDevice = vkDevice_->GetLogical();

	//logicalDevice.waitForFences(fences_[currentFrame_], true, std::numeric_limits<uint64_t>::max());
	//logicalDevice.resetFences(fences_[currentFrame_]);

	//vk::SubmitInfo submitInfo;

	//vk::Semaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
	//vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
	//submitInfo.waitSemaphoreCount = 1;
	//submitInfo.pWaitSemaphores = waitSemaphores;
	//submitInfo.pWaitDstStageMask = waitStages;

	//submitInfo.commandBufferCount = 1;
	//submitInfo.pCommandBuffers = &commandBuffers_[imageIndex];

	//vk::Semaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
	//submitInfo.signalSemaphoreCount = 1;
	//submitInfo.pSignalSemaphores = signalSemaphores;

	//vkDevice_->GetGraphicsQueue().submit(submitInfo, inFlightFences[currentFrame]);

	vk::PresentInfoKHR presentInfo;

	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &waitSemaphore;

	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &swapChain_;

	presentInfo.pImageIndices = &activeImageIndex_;

	presentQueue.presentKHR(presentInfo);

	currentFrame_ = (currentFrame_ + 1) % frameMax_;

}
