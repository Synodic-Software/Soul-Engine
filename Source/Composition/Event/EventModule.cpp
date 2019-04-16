#include "EventModule.h"

#include "Modules/Default/DefaultEventBackend.h"


// TODO: There will only ever be one DisplayModule system per Soul application. This will need to be
// moved to the build system per platform
std::unique_ptr<EventModule> EventModule::CreateModule()
{

	return std::make_unique<DefaultEventBackend>();

}