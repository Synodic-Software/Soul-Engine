#include "FiberProperties.h"

FiberProperties::FiberProperties(boost::fibers::context* context) :
	fiber_properties(context),
	priority(FiberPriority::LOW),
	runOnMain(false)
{
}

FiberPriority FiberProperties::GetPriority() const {
	return priority;
}

bool FiberProperties::RunOnMain() const {
	return runOnMain;
}

void FiberProperties::SetProperties(FiberPriority p, bool m) {
	if (p != priority || m != runOnMain) {
		priority = p;
		runOnMain = m;
		notify();
	}
}

