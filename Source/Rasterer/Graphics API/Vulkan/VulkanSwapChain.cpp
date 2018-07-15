#include "VulkanSwapChain.h"

#include "Core/Utility/Types.h"
#include "GLFW/glfw3.h"

VulkanSwapChain::VulkanSwapChain(std::shared_ptr<vk::Instance>& instance, std::vector<vk::PhysicalDevice>& physicalDevices, std::vector<char const*>& requiredExtensions, std::any& windowContext, glm::uvec2& size) :
	vulkanInstance_(instance),
	vSync(false)
{

	//GLFW uses vulkan c version
	VkSurfaceKHR castSurface;

	//garunteed to use GLFW if using vulkan (I hope this is future proof)
	const VkResult error = glfwCreateWindowSurface(
		static_cast<VkInstance>(*vulkanInstance_),
		std::any_cast<GLFWwindow*>(windowContext), //TODO: abstract the context
		nullptr,
		&castSurface
	);

	//back to c++ land
	surface_ = static_cast<vk::SurfaceKHR>(castSurface);

	assert(error == VK_SUCCESS);

	//TODO: abstract devices
	//TODO: proper device picking (based on a shared abstraction with Compute)

	//TODO: Deal with different queues for present + graphics
	int pickedDeviceIndex = -1;
	int pickedFamilyIndex = -1;

	//iterate over all devices to investigate their queue families for specific properties
	uint physicalDeviceIndex = 0;
	for (const auto& physicalDevice : physicalDevices) {

		std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

		pickedFamilyIndex = -1;

		//iterate over all queue properties to find the right one
		uint queueFamilyIndex = 0;
		for (const auto& queueFamilyProperty : queueFamilyProperties) {

			//the properties we want
			if (queueFamilyProperty.queueFlags & vk::QueueFlagBits::eGraphics && physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(queueFamilyIndex), surface_)) {
				pickedFamilyIndex = queueFamilyIndex;
				break;
			}

			++queueFamilyIndex;
		}

		//TODO does support swapchain?
		//conditions met to pick this physical device for the logical device
		if (pickedFamilyIndex < queueFamilyProperties.size()) {
			pickedDeviceIndex = physicalDeviceIndex;
			break;
		}

		++physicalDeviceIndex;
	}

	//TODO: deal with families + device not found
	assert(pickedDeviceIndex >= 0);
	assert(pickedFamilyIndex >= 0);

	vk::PhysicalDevice& physicalDevice = physicalDevices[pickedDeviceIndex];

	//setup the device with the given index
	float queuePriority = 0.0f;
	vk::DeviceQueueCreateInfo deviceQueueCreateInfo;
	deviceQueueCreateInfo.flags = vk::DeviceQueueCreateFlags();
	deviceQueueCreateInfo.queueFamilyIndex = static_cast<uint32_t>(pickedFamilyIndex);
	deviceQueueCreateInfo.queueCount = 1;
	deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

	vk::DeviceCreateInfo deviceCreateInfo;
	deviceCreateInfo.flags = vk::DeviceCreateFlags();
	deviceCreateInfo.queueCreateInfoCount = 1;
	deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
	deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
	deviceCreateInfo.ppEnabledExtensionNames = requiredExtensions.data();

	//TODO: dont push a new device every window. Manage with a global device list or something
	logicalDevice_ = physicalDevice.createDevice(
		deviceCreateInfo
	);


	//swapchain creation
	std::vector<vk::SurfaceFormatKHR> surfaceFormats = physicalDevice.getSurfaceFormatsKHR(surface_);

	assert(!surfaceFormats.empty());

	//TODO: proper colorspace selection
	const vk::Format format = surfaceFormats[0].format == vk::Format::eUndefined ? vk::Format::eB8G8R8A8Unorm : surfaceFormats[0].format;
	colorSpace_ = surfaceFormats[0].colorSpace;

	const vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface_);
	std::vector<vk::PresentModeKHR> presentModes = physicalDevice.getSurfacePresentModesKHR(surface_);

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
	swapchainCreateInfo.surface = surface_;
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

	swapChain_ = logicalDevice_.createSwapchainKHR(swapchainCreateInfo);


	vk::ImageViewCreateInfo colorAttachmentCreateInfo;
	colorAttachmentCreateInfo.format = format;
	colorAttachmentCreateInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
	colorAttachmentCreateInfo.subresourceRange.levelCount = 1;
	colorAttachmentCreateInfo.subresourceRange.layerCount = 1;
	colorAttachmentCreateInfo.viewType = vk::ImageViewType::e2D;

	auto swapChainImages = logicalDevice_.getSwapchainImagesKHR(swapChain_);

	images_.resize(swapChainImages.size());
	for (uint32_t i = 0; i < swapChainImages.size(); i++) {
		images_[i].image = swapChainImages[i];
		colorAttachmentCreateInfo.image = swapChainImages[i];
		images_[i].view = logicalDevice_.createImageView(colorAttachmentCreateInfo);
		images_[i].fence = vk::Fence();
	}

}

VulkanSwapChain::~VulkanSwapChain() {

	for (const auto& image : images_) {
		logicalDevice_.destroyImageView(image.view);
	}

	logicalDevice_.destroySwapchainKHR(swapChain_);

	vulkanInstance_->destroySurfaceKHR(surface_);

}


void VulkanSwapChain::Resize(glm::uvec2) {

}
