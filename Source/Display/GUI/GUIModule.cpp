#include "GUIModule.h"

#include "Core/System/Platform.h"
#include "Modules/Imgui/ImguiBackend.h"


// TODO: There will only ever be one DisplayModule system per Soul application. This will need to be
// moved to the build system per platform
std::shared_ptr<GUIModule> GUIModule::CreateModule()
{

	if constexpr (Platform::IsDesktop()) {
		return std::make_shared<ImguiBackend>();
	}
	else {
		return nullptr;
	}
}