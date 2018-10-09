#include "Graph.h"

#include "Parallelism/Fiber/Scheduler.h"


Graph::Graph(Scheduler* scheduler) :
	scheduler_(scheduler)
{

}

//Returns after dispatching the graph.
//Call scheduler_->Block() to guarantee completion
void Graph::Execute() {

	assert(!children_.empty());

	for (const auto& child : children_) {

		child->Execute();

	}

}

//Create a sub-graph
Graph& Graph::AddGraph() {

	return graphs_.emplace_front(scheduler_);

}