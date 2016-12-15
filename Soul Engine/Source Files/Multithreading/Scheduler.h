#pragma once

#include "Engine Core\BasicDependencies.h"

// tasks are submitted to scheduler
//  scheduler creates fiber
// which can be paused or executed on the core
namespace Scheduler{
	void Init();
	void Terminate();
}
