#pragma once

#include "Rasterer/RasterDevice.h"
#include "Parallelism/Thread/ThreadLocal.h"

#include <vulkan/vulkan.hpp>

class FiberScheduler;

struct SurfaceFormat
{
	
	vk::ColorSpaceKHR colorSpace;
	vk::Format colorFormat;

};

class VulkanDevice final: public RasterDevice {

public:

	VulkanDevice(std::shared_ptr<FiberScheduler>&, vk::PhysicalDevice&);
	~VulkanDevice() override;

	VulkanDevice(const VulkanDevice &) = delete;
	VulkanDevice(VulkanDevice &&) = default; //noexcept dependent on vk::device

	VulkanDevice& operator=(const VulkanDevice &) = delete;
	VulkanDevice& operator=(VulkanDevice &&) = default; //noexcept dependent on vk::device

	void Synchronize() override;

	const vk::Device& GetLogical() const;
	const vk::PhysicalDevice& GetPhysical() const;
	const vk::CommandPool& GetCommandPool() const;
	const vk::Queue& GetGraphicsQueue() const;
	const vk::Queue& GetPresentQueue() const;
	int GetGraphicsIndex() const;
	SurfaceFormat GetSurfaceFormat(const vk::SurfaceKHR&) const;

private:

	std::shared_ptr<FiberScheduler> scheduler_;

	std::vector<vk::Device> logicalDevices_; //noexcept dependent on vk::device
	vk::PhysicalDevice physicalDevice_;

	ThreadLocal<vk::CommandPool> commandPool_;

	vk::PhysicalDeviceProperties deviceProperties_;
	vk::PhysicalDeviceFeatures deviceFeatures_;
	vk::PhysicalDeviceMemoryProperties memoryProperties_;

	//TODO: refactor queue 
	vk::Queue graphicsQueue_;
	int graphicsIndex_; //TODO: refactor for index storage

};
