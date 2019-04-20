
##############################################
#Parallelism

target_sources(${PROJECT_NAME}
    PRIVATE
        Graph/Graph.cpp
        Graph/Graph.h
        Graph/Node.cpp
        Graph/Node.h
        Graph/Task.cpp
        Graph/Task.h
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
