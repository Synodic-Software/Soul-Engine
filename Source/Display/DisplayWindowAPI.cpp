#include "DisplayWindowAPI.h"

#include "Display/Window/GLFW/GLFWModule.h"
#include "System/Platform.h"

std::shared_ptr<DisplayWindowAPI> DisplayWindowAPI::CreateModule()
{
	if constexpr(Platform::IsDesktop())
	{
		return std::make_shared<GLFWModule>();
	}
	else {
		return nullptr;
	}
}