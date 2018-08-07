#pragma once

#include "Rasterer/Graphics API/Surface.h"

#include <vulkan/vulkan.hpp>
#include <any>
#include "Composition/Component/Component.h"

class VulkanContext;

class VulkanSurface : public Surface,  Component<VulkanSurface>{

public:

	VulkanSurface(VulkanContext*, std::any&);
	~VulkanSurface() override = default;

	VulkanSurface(const VulkanSurface&) = delete;
	VulkanSurface(VulkanSurface&& o) noexcept = default;

	VulkanSurface& operator=(const VulkanSurface&) = delete;
	VulkanSurface& operator=(VulkanSurface&& other) noexcept = default;

	vk::SurfaceKHR& GetSurface();

	void Terminate() override;

private:

	VulkanContext* context_;
	vk::SurfaceKHR surface_;

};
