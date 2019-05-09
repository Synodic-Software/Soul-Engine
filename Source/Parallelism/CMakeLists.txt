
##############################################
#Parallelism

target_sources(${PROJECT_NAME}
    PRIVATE
        Graph/Modules/EntityGraph/Graph.cpp
        Graph/Modules/EntityGraph/Graph.h
        Graph/Modules/EntityGraph/GraphNode.cpp
        Graph/Modules/EntityGraph/GraphNode.h
        Graph/Modules/EntityGraph/GraphTask.cpp
        Graph/Modules/EntityGraph/GraphTask.h
        Graph/GraphTask.h
        Graph/Graph.h
        Scheduler/Modules/Dispatch/DispatchSchedulerBackend.cpp
        Scheduler/Modules/Dispatch/DispatchSchedulerBackend.h
        Scheduler/Modules/Fiber/FiberProperties.cpp
        Scheduler/Modules/Fiber/FiberProperties.h
        Scheduler/Modules/Fiber/FiberSchedulerAlgorithm.cpp
        Scheduler/Modules/Fiber/FiberSchedulerAlgorithm.h
        Scheduler/Modules/Fiber/FiberSchedulerBackend.cpp
        Scheduler/Modules/Fiber/FiberSchedulerBackend.h
        Scheduler/SchedulerModule.cpp
        Scheduler/SchedulerModule.h
        Scheduler/TaskParameters.cpp
        Scheduler/TaskParameters.h
)
