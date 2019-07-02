#pragma once

#include "Pipeline/VulkanPipeline.h"
#include "VulkanFrameBuffer.h"
#include "Command/VulkanCommandBuffer.h"

#include <vulkan/vulkan.hpp>

class VulkanDevice;
class VulkanSurface;

class VulkanSwapChain {

public:

	VulkanSwapChain(std::unique_ptr<VulkanDevice>&,
		const VulkanSurface&, 
		bool,
		VulkanSwapChain* = nullptr);
	~VulkanSwapChain();

	VulkanSwapChain(const VulkanSwapChain&) = delete;
	VulkanSwapChain(VulkanSwapChain&& o) noexcept = default;

	VulkanSwapChain& operator=(const VulkanSwapChain&) = delete;
	VulkanSwapChain& operator=(VulkanSwapChain&& other) noexcept = default;

	void AquireImage();

	vk::Extent2D GetSize();


private:


	vk::Device device_;

	std::vector<vk::Image> renderBuffers_;
	std::vector<vk::ImageView> renderBufferViews_;

	uint currentFrame_;
	uint activeImageIndex_;
	uint frameMax_;

	vk::Extent2D size_;
	vk::SwapchainKHR swapChain_;

	std::vector<vk::Fence> frameFences_;
	std::vector<vk::Semaphore> presentSemaphores_;
	std::vector<vk::Semaphore> renderSemaphores_;


};
