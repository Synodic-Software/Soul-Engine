#pragma once

#include "Types.h"

#include <string>
#include <vulkan/vulkan.hpp>

class VulkanPhysicalDevice;


class VulkanInstance {

public:

	VulkanInstance(const vk::ApplicationInfo&, std::vector<std::string>, std::vector<std::string>);
	~VulkanInstance();

	VulkanInstance(const VulkanInstance&) = default;
	VulkanInstance(VulkanInstance&&) noexcept = default;

	VulkanInstance& operator=(const VulkanInstance&) = default;
	VulkanInstance& operator=(VulkanInstance&&) noexcept = default;

	const vk::Instance& Get();

private:

	vk::Instance instance_;
	std::vector<VulkanPhysicalDevice> physicalDevices_;

	// Dynamic dispatcher for extensions
	vk::DispatchLoaderDynamic dispatcher_;

	// Debug state
	// TODO: Should be conditionally included when the class is only debug mode.

	static VkBool32 DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT,
		VkDebugUtilsMessageTypeFlagsEXT,
		const VkDebugUtilsMessengerCallbackDataEXT*,
		void*);
	 
	vk::DebugUtilsMessengerEXT debugMessenger_;

};
