#include "MockWindowBackend.h"

MockWindowBackend::MockWindowBackend(std::shared_ptr<InputModule>& inputModule):
	WindowModule(inputModule)
{
}

void MockWindowBackend::Update()
{
}

void MockWindowBackend::Draw()
{
	
}

bool MockWindowBackend::Active()
{
	return true;
}

void MockWindowBackend::CreateWindow(const WindowParameters&, std::shared_ptr<RasterModule>&)
{
	
}

std::vector<const char*> MockWindowBackend::GetRasterExtensions()
{

	return std::vector<const char*>();

}
