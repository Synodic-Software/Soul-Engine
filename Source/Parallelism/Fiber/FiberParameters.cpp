#include "FiberParameters.h"

FiberParameters::FiberParameters():
	attach(true),
	swap(false),
	needsMainThread(false),
	priority(FiberPriority::LOW)
{
}

