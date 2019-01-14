#pragma once

#include <vulkan/vulkan.hpp>
#include <any>
#include "Composition/Component/Component.h"

class VulkanRasterBackend;

class VulkanSurface : Component<VulkanSurface>{

public:

	VulkanSurface(VulkanRasterBackend*, std::any&);
	~VulkanSurface() override;

	VulkanSurface(const VulkanSurface&) = delete;
	VulkanSurface(VulkanSurface&& o) noexcept = default;

	VulkanSurface& operator=(const VulkanSurface&) = delete;
	VulkanSurface& operator=(VulkanSurface&& other) noexcept = default;

	vk::SurfaceKHR& GetSurface();

private:

	VulkanRasterBackend* context_;
	vk::SurfaceKHR surface_;

};
