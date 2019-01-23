#include "FiberProperties.h"

FiberProperties::FiberProperties(boost::fibers::context* context) :
	fiber_properties(context),
	priority_(FiberPriority::HIGH),
	requiredThread_(-1)
{
}

FiberPriority FiberProperties::GetPriority() const {
	return priority_;
}

int FiberProperties::RequiredThread() const {
	return requiredThread_;
}

void FiberProperties::SetProperties(FiberPriority p, int m) {
	if (p != priority_ || m != requiredThread_) {
		priority_ = p;
		requiredThread_ = m;
		notify();
	}
}

