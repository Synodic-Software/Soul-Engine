#pragma once

#include "VulkanFramebuffer.h"
#include "VulkanSemaphore.h"
#include "VulkanFence.h"

#include <optional>

class VulkanFrame{

public:

	VulkanFrame() = default;
	~VulkanFrame() = default;

	VulkanFrame(const VulkanFrame&) = delete;
	VulkanFrame(VulkanFrame&& o) noexcept = default;

	VulkanFrame& operator=(const VulkanFrame&) = delete;
	VulkanFrame& operator=(VulkanFrame&& other) noexcept = default;

	VulkanFrameBuffer& Framebuffer();
	
private:
	
	std::optional<VulkanFrameBuffer> framebuffer_;
	std::optional<VulkanSemaphore> presentSemaphore_;
	std::optional<VulkanSemaphore> renderSemaphore_;
	std::optional<VulkanFence> imageFence_;
	
};
