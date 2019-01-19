#pragma once

//#include "VulkanPipeline.h"
//#include "VulkanFramebuffer.h"

#include "Composition/Component/Component.h"

#include <vulkan/vulkan.hpp>
#include <glm/vec2.hpp>

//class VulkanDevice;
//class VulkanSurface;
//class EntityManager;


struct SwapChainSurface {
	vk::Image image;
	vk::ImageView view;
	vk::Fence fence;
};

class VulkanSwapChain {

public:

	VulkanSwapChain(EntityManager*, vk::SurfaceKHR&, glm::uvec2&, bool, VulkanSwapChain* = nullptr);
	~VulkanSwapChain();

	VulkanSwapChain(const VulkanSwapChain&) = delete;
	VulkanSwapChain(VulkanSwapChain&& o) noexcept = default;

	VulkanSwapChain& operator=(const VulkanSwapChain&) = delete;
	VulkanSwapChain& operator=(VulkanSwapChain&& other) noexcept = default;


private:

	vk::SwapchainKHR swapChain_;

	std::vector<SwapChainSurface> surfaces_;



};
