#include "Graph.h"

#include "Parallelism/Fiber/Scheduler.h"


Graph::Graph(Scheduler& scheduler):
	scheduler_(scheduler)
{
	
}


void Graph::Execute() {

	for (const auto& task : tasks_) {

		task.Execute(scheduler_);

	}

}