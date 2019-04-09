#include "GUIModule.h"

#include "System/Platform.h"
#include "Modules/Imgui/ImguiBackend.h"


// TODO: There will only ever be one DisplayModule system per Soul application. This will need to be
// moved to the build system per platform
std::unique_ptr<GUIModule> GUIModule::CreateModule()
{

	if constexpr (Platform::IsDesktop()) {
		return std::make_unique<ImguiBackend>();
	}
	else {
		return nullptr;
	}
}