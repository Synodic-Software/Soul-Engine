#pragma once

#include "Types.h"
#include "Render/Raster/RasterDevice.h"
#include "Core/Structure/Span.h"

#include <map>
#include <vulkan/vulkan.hpp>

class SchedulerModule;
class VulkanQueue;

enum class QueueFamilyType {
	Compute, 
	Graphics,	//Can't be Compute
	Transfer	//Can't be Compute or Graphics
};

static constexpr uint queueFamilyTypeCount = 3;

class VulkanDevice final: public RasterDevice {

public:

	VulkanDevice(std::shared_ptr<SchedulerModule>&,
		const vk::PhysicalDevice&,
		nonstd::span<std::string>,
		nonstd::span<std::string>);
	~VulkanDevice() override;

	VulkanDevice(const VulkanDevice &) = delete;
	VulkanDevice(VulkanDevice &&) noexcept = default;

	VulkanDevice& operator=(const VulkanDevice &) = delete;
	VulkanDevice& operator=(VulkanDevice&&) noexcept = default;

	void Synchronize() override;

	const vk::Device& GetLogical() const;
	const vk::PhysicalDevice& GetPhysical() const;


private:

	typedef struct QueueMember {

		uint familyIndex;
		uint count;

		QueueMember(uint familyIndexIn, uint countIn): familyIndex(familyIndexIn), count(countIn)
		{
		}

		bool operator==(const uint32_t& familyIndexIn) const
		{
			return familyIndex == familyIndexIn;
		}
	};

	typedef struct DeviceQueueFamilyInfo {

		uint familyQueueCount[queueFamilyTypeCount];
		std::map<QueueFamilyType, std::vector<QueueMember>> queueFamily;

	};

	std::shared_ptr<SchedulerModule> scheduler_;

	std::vector<vk::Device> logicalDevices_;
	vk::PhysicalDevice physicalDevice_;

	vk::PhysicalDeviceProperties deviceProperties_;
	vk::PhysicalDeviceFeatures deviceFeatures_;
	vk::PhysicalDeviceMemoryProperties memoryProperties_;

	DeviceQueueFamilyInfo familyInfo_;
	std::vector<VulkanQueue> queues_;

};
