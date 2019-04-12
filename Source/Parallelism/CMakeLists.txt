
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
        Modules/Fiber/FiberParameters.cpp
        Modules/Fiber/FiberParameters.h
        Modules/Fiber/FiberProperties.cpp
        Modules/Fiber/FiberProperties.h
        Modules/Fiber/FiberScheduler.cpp
        Modules/Fiber/FiberScheduler.h
        Modules/Fiber/FiberSchedulerAlgorithm.cpp
        Modules/Fiber/FiberSchedulerAlgorithm.h
        Modules/Mock/MockScheduler.cpp
        Modules/Mock/MockScheduler.h
        Scheduler.cpp
        Scheduler.h
        Thread/ThreadLocal.cpp
        Thread/ThreadLocal.h
)
