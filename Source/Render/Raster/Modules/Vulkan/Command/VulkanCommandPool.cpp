#include "VulkanCommandPool.h"

#include "Parallelism/Scheduler/SchedulerModule.h"
#include "Render/Raster/Modules/Vulkan/Device/VulkanDevice.h"

VulkanCommandPool::VulkanCommandPool(std::shared_ptr<SchedulerModule>& scheduler,
	const vk::Device& vulkanDevice,
	uint queueFamilyIndex) :
	scheduler_(scheduler),
	device_(vulkanDevice)
{

	vk::CommandPoolCreateInfo poolInfo;
	poolInfo.flags = vk::CommandPoolCreateFlagBits::eTransient |
					 vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	poolInfo.queueFamilyIndex = queueFamilyIndex;

	// TODO: move to 3 commandpools per thread as suggested by NVIDIA
	scheduler_->ForEachThread(TaskPriority::UX, [&]() {
		// TODO: multiple logical devices
		commandPool_ = device_.createCommandPool(poolInfo);
	});

}

VulkanCommandPool::~VulkanCommandPool()
{

	scheduler_->ForEachThread(TaskPriority::UX, [&]() noexcept {
		// TODO: multiple logical devices
		device_.destroyCommandPool(commandPool_);
	});

}


const vk::CommandPool& VulkanCommandPool::GetCommandPool() const
{

	return commandPool_;

}