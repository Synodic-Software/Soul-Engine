#include "InputModule.h"
#include "Core/System/Platform.h"

#include "Modules/GLFW/GLFWInputBackend.h"


//TODO: There will only ever be one DisplayModule system per Soul application. This will need to be moved to the build system per platform
std::unique_ptr<InputModule> InputModule::CreateModule()
{

	if constexpr (Platform::IsDesktop()) {
		return std::make_unique<GLFWInputBackend>();
	}
	else {
		return nullptr;
	}

}