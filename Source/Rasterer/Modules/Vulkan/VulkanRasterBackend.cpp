#include "VulkanRasterBackend.h"

#include "VulkanDevice.h"
#include "Core/Utility/Exception/Exception.h"

void VulkanRasterBackend::Draw()
{

	throw NotImplemented();

}

void VulkanRasterBackend::DrawIndirect()
{

	throw NotImplemented();

}

std::shared_ptr<RasterDevice> VulkanRasterBackend::CreateDevice()
{
	return std::shared_ptr<VulkanDevice>();
}
