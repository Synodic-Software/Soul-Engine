#include "MockRasterBackend.h"

#include "MockDevice.h"
#include "Core/Utility/Exception/Exception.h"

void MockRasterBackend::Draw()
{

	throw NotImplemented();

}

void MockRasterBackend::DrawIndirect()
{

	throw NotImplemented();

}

std::shared_ptr<RasterDevice> MockRasterBackend::CreateDevice()
{

	return std::shared_ptr<MockDevice>();

}
