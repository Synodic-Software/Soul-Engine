#pragma once

#include "Rasterer/RasterBackend.h"

#include <vulkan/vulkan.hpp>

class Display;

class VulkanRasterBackend final : public RasterBackend {

public:

	VulkanRasterBackend(std::shared_ptr<Display>);
	~VulkanRasterBackend() override = default;

	VulkanRasterBackend(const VulkanRasterBackend &) = delete;
	VulkanRasterBackend(VulkanRasterBackend &&) noexcept = default;

	VulkanRasterBackend& operator=(const VulkanRasterBackend &) = delete;
	VulkanRasterBackend& operator=(VulkanRasterBackend &&) noexcept = default;

	void Draw() override;
	void DrawIndirect() override;

	std::shared_ptr<RasterDevice> CreateDevice() override;

	vk::Instance& GetInstance();

private:


	std::vector<char const*> requiredDeviceExtensions_;
	std::vector<char const*> requiredInstanceExtensions_;
	std::vector<const char*> validationLayers_;

	vk::ApplicationInfo appInfo_;
	vk::Instance instance_;
	std::vector<vk::PhysicalDevice> physicalDevices_;

	//Debug related state 
	//TODO: Should be conditionally included when the class is only debug mode.

	static VkBool32 DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT*, void*);

	VkDebugUtilsMessengerEXT debugMessenger_;
	vk::DispatchLoaderDynamic dispatcher_;
	vk::DebugReportCallbackEXT callback_;

};