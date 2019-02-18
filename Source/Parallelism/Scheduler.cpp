#include "Scheduler.h"

#include "Modules/Fiber/FiberScheduler.h"

//TODO: This needs to instead be runtime loadable from shared libraries or statically linked
std::shared_ptr<FiberScheduler> Scheduler::CreateModule(Property<uint>& threadCount)
{

	return std::make_shared<FiberScheduler>(threadCount);

}
