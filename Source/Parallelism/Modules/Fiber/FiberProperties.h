#pragma once

#include "boost/fiber/properties.hpp"
#include "FiberParameters.h"

class FiberProperties : public boost::fibers::fiber_properties {

public:

	//Construction
	FiberProperties(boost::fibers::context*);

	//Implementation
	FiberPriority GetPriority() const;
	int RequiredThread() const;

	void SetProperties(FiberPriority, int);

private:

	FiberPriority priority_;
	int requiredThread_;

};
