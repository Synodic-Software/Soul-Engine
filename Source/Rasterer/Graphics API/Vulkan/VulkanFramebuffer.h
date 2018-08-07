#pragma once

#include "Rasterer/Graphics API/Framebuffer.h"

class VulkanFramebuffer : public Framebuffer {

public:

	VulkanFramebuffer() = default;
	virtual ~VulkanFramebuffer() = default;

	VulkanFramebuffer(const VulkanFramebuffer&) = delete;
	VulkanFramebuffer(VulkanFramebuffer&& o) noexcept = delete;

	VulkanFramebuffer& operator=(const VulkanFramebuffer&) = delete;
	VulkanFramebuffer& operator=(VulkanFramebuffer&& other) noexcept = delete;

};
