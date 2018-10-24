#include "Graph.h"

#include "Parallelism/Fiber/Scheduler.h"


Graph::Graph(Scheduler* scheduler) :
	scheduler_(scheduler)
{

}

//Returns after dispatching the graph.
//Call scheduler_->Block() to guarantee completion
void Graph::Execute(std::chrono::nanoseconds targetDuration) {

	for (const auto& child : children_) {

		if (child->Root()) {

			child->Execute();

		}

	}

}

//Create a sub-graph
Graph& Graph::AddGraph() {

	return graphs_.emplace_front(scheduler_);

}