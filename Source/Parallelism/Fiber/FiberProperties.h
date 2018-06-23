#pragma once

#include "boost/fiber/properties.hpp"
#include "FiberParameters.h"

class FiberProperties : public boost::fibers::fiber_properties {

public:

	//Construction
	FiberProperties(boost::fibers::context*);

	//Implementation
	FiberPriority GetPriority() const;
	bool RunOnMain() const;

	void SetProperties(FiberPriority, bool);
	

private:

	FiberPriority priority;
	bool runOnMain;
};
