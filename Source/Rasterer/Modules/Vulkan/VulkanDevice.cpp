#include "VulkanRasterBackend.h"

#include "VulkanDevice.h"

std::shared_ptr<RasterDevice> VulkanRasterBackend::CreateDevice()

{
	return std::shared_ptr<VulkanDevice>();

}
