#pragma once

#include "Rasterer/RasterBackend.h"

#include "VulkanSwapChain.h"

#include <vulkan/vulkan.hpp>
#include <glm/vec2.hpp>


class FiberScheduler;
class DisplayModule;
class VulkanDevice;
class VulkanSwapChain;

class VulkanRasterBackend final : public RasterBackend {

public:

	VulkanRasterBackend(std::shared_ptr<FiberScheduler>&, std::shared_ptr<DisplayModule>&);
	~VulkanRasterBackend() override;

	VulkanRasterBackend(const VulkanRasterBackend &) = delete;
	VulkanRasterBackend(VulkanRasterBackend &&) noexcept = default;

	VulkanRasterBackend& operator=(const VulkanRasterBackend &) = delete;
	VulkanRasterBackend& operator=(VulkanRasterBackend &&) noexcept = default;

	void Draw() override;
	void DrawIndirect() override;

	std::unique_ptr<VulkanSwapChain> RegisterSurface(vk::SurfaceKHR&, glm::uvec2, VulkanSwapChain* = nullptr);
	void RemoveSurface(vk::SurfaceKHR&);

	void AddInstanceExtensions(std::vector<char const*>&);
	vk::Instance& GetInstance();


private:

	std::vector<char const*> requiredInstanceExtensions_;
	std::vector<const char*> validationLayers_;

	vk::Instance instance_;
	std::vector<std::shared_ptr<VulkanDevice>> devices_;

	//Dynamic dispatcher for extensions
	vk::DispatchLoaderDynamic dispatcher_;

	//Debug state 
	//TODO: Should be conditionally included when the class is only debug mode.

	static VkBool32 DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT*, void*);

	vk::DebugUtilsMessengerEXT debugMessenger_;
	vk::DebugReportCallbackEXT callback_;

};