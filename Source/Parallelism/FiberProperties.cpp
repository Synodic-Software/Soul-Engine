#include "FiberProperties.h"

FiberProperties::FiberProperties(boost::fibers::context * context) :
	fiber_properties(context),
	priority(0),
	runOnMain(false)
{
}

int FiberProperties::GetPriority() const {
	return priority;
}

bool FiberProperties::RunOnMain() const {
	return runOnMain;
}

void FiberProperties::SetPriority(std::thread::id id, int p, bool m) {
	if (id != mainID || p != priority || m != runOnMain) {
		mainID = id;
		priority = p;
		runOnMain = m;
		notify();
	}
}

