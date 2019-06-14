#include "VulkanInstance.h"

VulkanInstance::VulkanInstance()
{
}

VulkanInstance::~VulkanInstance()
{
}

const vk::Instance& VulkanInstance::Get()
{

	return instance_;

}