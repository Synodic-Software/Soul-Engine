#pragma once

#include "Pipeline/VulkanPipeline.h"
#include "VulkanFrameBuffer.h"
#include "Command/VulkanCommandBuffer.h"
#include "Core/Composition/Component/Component.h"

#include <vulkan/vulkan.hpp>

class VulkanDevice;
class VulkanSurface;

class VulkanSwapChain : Component {

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

	nonstd::span<vk::Image> Images();
	nonstd::span<vk::ImageView> ImageViews();
	uint ActiveImageIndex() const;

	void AquireImage(const vk::Semaphore&);

	vk::Extent2D Size() const;


private:

	vk::Device device_;

	std::vector<vk::Image> renderImages_;
	std::vector<vk::ImageView> renderImageViews_;

	uint activeImageIndex_;

	vk::Extent2D size_;
	vk::SwapchainKHR swapChain_;


};
