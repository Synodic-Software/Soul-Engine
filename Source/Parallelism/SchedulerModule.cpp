#include "SchedulerModule.h"

#include "Modules/Fiber/FiberSchedulerBackend.h"

//TODO: This needs to instead be runtime loadable from shared libraries or statically linked
std::shared_ptr<SchedulerModule> SchedulerModule::CreateModule(Property<uint>& threadCount)
{

	return std::make_shared<FiberSchedulerBackend>(threadCount);

}
