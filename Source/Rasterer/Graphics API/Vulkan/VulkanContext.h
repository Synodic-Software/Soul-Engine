#pragma once

#include "Rasterer/Graphics API/RasterContext.h"
#include "Composition/Entity/EntityManager.h"

#include "vulkan/vulkan.hpp"

#include <vector>

class Scheduler;
class RasterDevice;

class VulkanContext final : public RasterContext {

public:

	VulkanContext(Scheduler&, EntityManager&);
	~VulkanContext() override;

	VulkanContext(const VulkanContext&) = delete;
	VulkanContext(VulkanContext&&) noexcept = delete;

	VulkanContext& operator=(const VulkanContext&) = delete;
	VulkanContext& operator=(VulkanContext&&) noexcept = delete;

	Entity CreateSurface(std::any&) override;
	std::unique_ptr<SwapChain> CreateSwapChain(Entity, Entity, glm::uvec2&) override;
	Entity CreateDevice(Entity) override;

	const vk::Instance& GetInstance() const;
	const std::vector<vk::PhysicalDevice>& GetPhysicalDevices() const;

	void Synchronize() override;

private:

	VkDebugUtilsMessengerEXT debugMessenger_;
	vk::DispatchLoaderDynamic dispatcher_;

	static VkBool32 DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT*, void*);

	Scheduler& scheduler_;
	EntityManager& entityManager_;

	vk::Instance instance_;

	//TODO alternate storage
	std::vector<char const*> requiredDeviceExtensions_;
	std::vector<char const*> requiredInstanceExtensions_;

	std::vector<vk::PhysicalDevice> physicalDevices_;
	std::vector<Entity> logicalDevices_;

#ifdef NDEBUG

	static constexpr bool validationEnabled_ = false;

#else

	static constexpr bool validationEnabled_ = true;

	vk::DebugReportCallbackEXT callback_;

#endif

};