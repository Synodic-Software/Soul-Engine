#include "VulkanCommandPool.h"

#include "Parallelism/Scheduler/SchedulerModule.h"
#include "Render/Raster/Modules/Vulkan/Device/VulkanDevice.h"

VulkanCommandPool::VulkanCommandPool(std::shared_ptr<SchedulerModule>& scheduler,
	const std::shared_ptr<VulkanDevice>& vulkanDevice):
	scheduler_(scheduler),
	device_(vulkanDevice)
{

	const vk::Device& logicalDevice = device_->GetLogical();

	vk::CommandPoolCreateInfo poolInfo;
	poolInfo.flags = vk::CommandPoolCreateFlagBits::eTransient |
					 vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	poolInfo.queueFamilyIndex = device_->GetGraphicsIndex();

	// TODO: move to 3 commandpools per thread as suggested by NVIDIA
	scheduler_->ForEachThread(TaskPriority::UX, [&]() {
		// TODO: multiple logical devices
		commandPool_ = logicalDevice.createCommandPool(poolInfo);
	});

}

VulkanCommandPool::~VulkanCommandPool()
{

	const vk::Device& logicalDevice = device_->GetLogical();

	scheduler_->ForEachThread(TaskPriority::UX, [&]() noexcept {
		// TODO: multiple logical devices
		logicalDevice.destroyCommandPool(commandPool_);
	});

}


const vk::CommandPool& VulkanCommandPool::GetCommandPool() const
{

	return commandPool_;

}