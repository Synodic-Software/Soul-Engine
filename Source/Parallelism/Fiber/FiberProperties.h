#pragma once

#include "boost/fiber/properties.hpp"
#include <thread>

class FiberProperties : public boost::fibers::fiber_properties {

public:

	//Construction
	FiberProperties(boost::fibers::context*);

	//Implementation
	int GetPriority() const;
	bool RunOnMain() const;
	void SetPriority(std::thread::id, int, bool);


private:

	int priority;
	bool runOnMain;
	std::thread::id mainID;

};
