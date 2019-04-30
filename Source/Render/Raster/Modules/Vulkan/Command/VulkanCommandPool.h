#pragma once

#include "Core/Utility/Thread/ThreadLocal.h"

#include <memory>
#include <vulkan/vulkan.hpp>

class SchedulerModule;
class VulkanDevice;

class VulkanCommandPool final {

public:

	VulkanCommandPool(std::shared_ptr<SchedulerModule>&, const std::shared_ptr<VulkanDevice>&);
	~VulkanCommandPool();

	VulkanCommandPool(const VulkanCommandPool&) = delete;
	VulkanCommandPool(VulkanCommandPool&&) noexcept = default;

	VulkanCommandPool& operator=(const VulkanCommandPool&) = delete;
	VulkanCommandPool& operator=(VulkanCommandPool&&) noexcept = default;

	const vk::CommandPool& GetCommandPool() const;


private:

	std::shared_ptr<SchedulerModule> scheduler_;
	std::shared_ptr<VulkanDevice> device_;

	ThreadLocal<vk::CommandPool> commandPool_;


};
