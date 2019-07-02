#include "VulkanSurface.h"

VulkanSurface::VulkanSurface(const vk::Instance& instance, const vk::SurfaceKHR& surface): 
	instance_(instance), surface_(surface)
{
}

VulkanSurface::~VulkanSurface()
{

	instance_.destroySurfaceKHR(surface_);

}

vk::SurfaceKHR VulkanSurface::Handle()
{
	return surface_;
}

SurfaceFormat VulkanSurface::Format(const vk::PhysicalDevice& physicalDevice) const
{



	const auto formats = physicalDevice.getSurfaceFormatsKHR(surface_);

	SurfaceFormat surfaceFormat = {
		vk::ColorSpaceKHR::eSrgbNonlinear,
		vk::Format::eB8G8R8A8Unorm,
	};

	// TODO: pick formats better
	if (!formats.empty() && formats.front().format == vk::Format::eUndefined) {
		return surfaceFormat;
	}

	for (const auto& format : formats) {

		if (format.format == vk::Format::eB8G8R8A8Unorm &&
			format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
			return surfaceFormat;
		}
	}

	surfaceFormat.colorFormat = formats.front().format;
	surfaceFormat.colorSpace = formats.front().colorSpace;

	return surfaceFormat;
}