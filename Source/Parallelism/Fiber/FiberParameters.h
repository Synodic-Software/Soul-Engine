#pragma once

enum class FiberPriority {
	LOW, //A comparitivley low priority task.
	HIGH, //A high priority task.	
	UX //User Experience. Takes precedence over all.
};

class FiberParameters {

public:

	FiberParameters(bool = true, FiberPriority = FiberPriority::HIGH, bool = true, int = -1);

private:

	friend class Scheduler;

	bool post_; // the fiber will not execute immediatly and instead run when a scheduler picks it up. If not true, the fiber is required to remain on this thread and other parameters dont matter.	
	FiberPriority priority_;
	bool shouldBlock_; //attaches the fiber to its parent. Garunteed to complete before the parent does and can synchronize with `Block()`
	int requiredThread_; //forces the fiber to run on a pecified thread, any negative number runs on any thread

};
