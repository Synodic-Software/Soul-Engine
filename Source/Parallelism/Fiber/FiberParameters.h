#pragma once

enum class FiberPriority {
	LOW, //A comparitivley low priority task.
	HIGH, //A high priority task.	
	UX //User Experience. Takes precedence over all.
};

class FiberParameters {

public:

	FiberParameters();

	bool attach; //attaches the fiber to its parent
	bool swap; // the fiber will be immediatly executed and the current fiber put onto queue
	bool needsMainThread; //forces the fiber to run on the main thread
	FiberPriority priority;
	
};
