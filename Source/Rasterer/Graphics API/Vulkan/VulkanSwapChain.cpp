#include "VulkanSwapChain.h"
#include "VulkanContext.h"
#include "VulkanSurface.h"
#include "VulkanDevice.h"

VulkanSwapChain::VulkanSwapChain(EntityManager& entityManager, Entity device, Entity surface, glm::uvec2& size) :
	entityManager_(entityManager),
	device_(device),
	vSync(false)
{

	const auto& vkDevice = entityManager_.GetComponent<VulkanDevice>(device_);
	const auto& logicalDevice = vkDevice.GetLogicalDevice();
	const auto& physicalDevice = vkDevice.GetPhysicalDevice();

	const auto& vkSurface = entityManager_.GetComponent<VulkanSurface>(surface).GetSurface();

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
	pipeline_ = std::make_unique<VulkanPipeline>(entityManager_, device_, swapchainSize, "../../Soul Engine/Resources/Shaders/vert.spv", "../../Soul Engine/Resources/Shaders/frag.spv", format);


	//TODO: framebuffers dont get deleted, transfer to ECS and manually terminate
	for (SwapChainImage& image: images_) {
		framebuffers_.emplace_back(entityManager_, device_, image.view, pipeline_->GetRenderPass(), size);
	}

}

VulkanSwapChain::~VulkanSwapChain() {

	const vk::Device& logicalDevice = entityManager_.GetComponent<VulkanDevice>(device_).GetLogicalDevice();

	for (const auto& image : images_) {
		logicalDevice.destroyImageView(image.view);
	}

	for (auto& framebuffer : framebuffers_) {
		framebuffer.Terminate();
	}

	logicalDevice.destroySwapchainKHR(swapChain_);

}


void VulkanSwapChain::Resize(glm::uvec2) {

}
