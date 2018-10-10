#include "VulkanSwapChain.h"
#include "VulkanContext.h"
#include "VulkanSurface.h"
#include "VulkanDevice.h"
#include "Composition/Entity/EntityManager.h"

VulkanSwapChain::VulkanSwapChain(EntityManager* entityManager, Entity device, Entity surface, glm::uvec2& size) :
	entityManager_(entityManager),
	device_(device),
	currentFrame(0),
	flightFramesCount(2),
	vSync(false)
{

	BuildSwapChain(surface, size, true);

}

void VulkanSwapChain::BuildSwapChain(Entity surface, glm::uvec2& size, bool createPipeline) {

	const auto& vkDevice = entityManager_->GetComponent<VulkanDevice>(device_);
	const auto& logicalDevice = vkDevice.GetLogicalDevice();
	const auto& physicalDevice = vkDevice.GetPhysicalDevice();

	const auto& vkSurface = entityManager_->GetComponent<VulkanSurface>(surface).GetSurface();

	//swapchain creation
	std::vector<vk::SurfaceFormatKHR> surfaceFormats = physicalDevice.getSurfaceFormatsKHR(vkSurface);

	assert(!surfaceFormats.empty());

	//TODO: proper colorspace selection
	const vk::Format format = surfaceFormats[0].format == vk::Format::eUndefined ? vk::Format::eB8G8R8A8Unorm : surfaceFormats[0].format;
	colorSpace_ = surfaceFormats[0].colorSpace;

	const vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(vkSurface);
	std::vector<vk::PresentModeKHR> presentModes = physicalDevice.getSurfacePresentModesKHR(vkSurface);

	vk::Extent2D swapchainSize;
	if (surfaceCapabilities.currentExtent.width == std::numeric_limits<uint32_t>::max())
	{
		swapchainSize.width = size.x;
		swapchainSize.height = size.y;
	}
	else
	{
		swapchainSize = surfaceCapabilities.currentExtent;
	}

	//TODO: abstract into present options
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


	//proper rotation prior to display
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
	swapchainCreateInfo.surface = vkSurface;
	swapchainCreateInfo.minImageCount = imageCount;
	swapchainCreateInfo.imageFormat = format;
	swapchainCreateInfo.imageColorSpace = colorSpace_;
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

	swapChain_ = logicalDevice.createSwapchainKHR(swapchainCreateInfo);

	vk::ImageViewCreateInfo colorAttachmentCreateInfo;
	colorAttachmentCreateInfo.format = format;
	colorAttachmentCreateInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
	colorAttachmentCreateInfo.subresourceRange.levelCount = 1;
	colorAttachmentCreateInfo.subresourceRange.layerCount = 1;
	colorAttachmentCreateInfo.viewType = vk::ImageViewType::e2D;

	auto swapChainImages = logicalDevice.getSwapchainImagesKHR(swapChain_);

	images_.resize(swapChainImages.size());
	for (uint32_t i = 0; i < swapChainImages.size(); ++i) {
		images_[i].image = swapChainImages[i];
		colorAttachmentCreateInfo.image = swapChainImages[i];
		images_[i].view = logicalDevice.createImageView(colorAttachmentCreateInfo);
		images_[i].fence = vk::Fence();
	}

	//TODO: Remove hardcoded pipeline + Hardcoded paths
	//TODO: Associate paths to Project/Executable
	if (createPipeline) {
		pipeline_ = std::make_unique<VulkanPipeline>(*entityManager_, device_, swapchainSize, "../../Soul Engine/Resources/Shaders/vert.spv", "../../Soul Engine/Resources/Shaders/frag.spv", format);
	} else {
		pipeline_->Create(swapchainSize, format, true);
	}

	for (SwapChainImage& image : images_) {
		frameBuffers_.emplace_back(*entityManager_, device_, image.view, pipeline_->GetRenderPass(), size);
	}

	vk::CommandBufferAllocateInfo allocInfo;
	allocInfo.commandPool = vkDevice.GetCommandPool();
	allocInfo.level = vk::CommandBufferLevel::ePrimary;
	allocInfo.commandBufferCount = static_cast<uint32_t>(swapChainImages.size());

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
		commandBuffers_[i].draw(3, 1, 0, 0);
		commandBuffers_[i].endRenderPass();

		commandBuffers_[i].end();

	}


	//set up synchronization primatives
	imageAvailableSemaphores.resize(flightFramesCount);
	renderFinishedSemaphores.resize(flightFramesCount);
	inFlightFences.resize(flightFramesCount);

	vk::SemaphoreCreateInfo semaphoreInfo;

	vk::FenceCreateInfo fenceInfo;
	fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

	for (size_t i = 0; i < flightFramesCount; i++) {

		imageAvailableSemaphores[i] = logicalDevice.createSemaphore(semaphoreInfo);
		renderFinishedSemaphores[i] = logicalDevice.createSemaphore(semaphoreInfo);

		inFlightFences[i] = logicalDevice.createFence(fenceInfo);

	}
}

void VulkanSwapChain::Terminate() {

	const auto& vkDevice = entityManager_->GetComponent<VulkanDevice>(device_);
	const auto& logicalDevice = vkDevice.GetLogicalDevice();

	logicalDevice.freeCommandBuffers(vkDevice.GetCommandPool(),
			static_cast<uint32_t>(commandBuffers_.size()), commandBuffers_.data());


	for (auto& framebuffer : frameBuffers_) {
		framebuffer.Terminate();
	}

	frameBuffers_.clear();

	for (const auto& image : images_) {
		logicalDevice.destroyImageView(image.view);
	}

	for (size_t i = 0; i < flightFramesCount; i++) {

		logicalDevice.destroySemaphore(imageAvailableSemaphores[i]);
		logicalDevice.destroySemaphore(renderFinishedSemaphores[i]);
		logicalDevice.destroyFence(inFlightFences[i]);

	}

	logicalDevice.destroySwapchainKHR(swapChain_);

}

void VulkanSwapChain::Resize(Entity surface, glm::uvec2 size) {

	Terminate();

	pipeline_->Terminate();
	pipeline_->GetRenderPass().Terminate();

	auto& vkDevice = entityManager_->GetComponent<VulkanDevice>(device_);
	vkDevice.Rebuild();

	BuildSwapChain(surface, size, false);

}

void VulkanSwapChain::Draw() {

	const auto& vkDevice = entityManager_->GetComponent<VulkanDevice>(device_);
	const auto& logicalDevice = vkDevice.GetLogicalDevice();

	logicalDevice.waitForFences(inFlightFences[currentFrame], true, std::numeric_limits<uint64_t>::max());
	logicalDevice.resetFences(inFlightFences[currentFrame]);

	auto[acquireResult, imageIndex] = logicalDevice.acquireNextImageKHR(swapChain_, std::numeric_limits<uint64_t>::max(), imageAvailableSemaphores[currentFrame], nullptr);

	//TODO: handle this occurrence
	/*if((VkResult) acquireResult != VK_SUCCESS) {
	 
	}*/

	vk::SubmitInfo submitInfo;

	vk::Semaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
	vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = waitSemaphores;
	submitInfo.pWaitDstStageMask = waitStages;

	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffers_[imageIndex];

	vk::Semaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = signalSemaphores;

	vkDevice.GetGraphicsQueue().submit(submitInfo, inFlightFences[currentFrame]);

	vk::PresentInfoKHR presentInfo;

	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = signalSemaphores;

	vk::SwapchainKHR swapChains[] = { swapChain_ };
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = swapChains;

	presentInfo.pImageIndices = &imageIndex;

	vkDevice.GetPresentQueue().presentKHR(presentInfo);

	currentFrame = (currentFrame + 1) % flightFramesCount;

}
