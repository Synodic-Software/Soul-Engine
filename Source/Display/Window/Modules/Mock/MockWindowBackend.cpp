#include "MockWindowBackend.h"

#include "MockWindow.h"
#include "Core/Utility/Exception/Exception.h"

MockWindowBackend::MockWindowBackend(std::shared_ptr<InputModule>& inputModule)
{
}

void MockWindowBackend::Update()
{

	throw NotImplemented();

}

bool MockWindowBackend::Active()
{

	throw NotImplemented();

}

void MockWindowBackend::CreateWindow(const WindowParameters&, std::shared_ptr<RasterModule>&)
{

	throw NotImplemented();

}

std::vector<const char*> MockWindowBackend::GetRasterExtensions()
{

	throw NotImplemented();

	return std::vector<const char*>();

}

Window& MockWindowBackend::GetWindow()
{

	throw NotImplemented();

	return MockWindow();

}