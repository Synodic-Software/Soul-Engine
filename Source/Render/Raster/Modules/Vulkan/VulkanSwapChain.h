#pragma once

#include "Pipeline/VulkanPipeline.h"
#include "VulkanFrameBuffer.h"
#include "Command/VulkanCommandBuffer.h"

#include <vulkan/vulkan.hpp>

class VulkanDevice;
class VulkanSurface;

class VulkanSwapChain {

public:

	VulkanSwapChain(VulkanDevice&,
		VulkanSurface&, 
		bool,
		VulkanSwapChain* = nullptr);
	~VulkanSwapChain();

	VulkanSwapChain(const VulkanSwapChain&) = delete;
	VulkanSwapChain(VulkanSwapChain&& o) noexcept = default;

	VulkanSwapChain& operator=(const VulkanSwapChain&) = delete;
	VulkanSwapChain& operator=(VulkanSwapChain&& other) noexcept = default;

	void AquireImage(const vk::Semaphore&);

	vk::Extent2D GetSize();


private:

	vk::Device device_;

	std::vector<vk::Image> renderBuffers_;
	std::vector<vk::ImageView> renderBufferViews_;

	uint activeImageIndex_;

	vk::Extent2D size_;
	vk::SwapchainKHR swapChain_;


};
