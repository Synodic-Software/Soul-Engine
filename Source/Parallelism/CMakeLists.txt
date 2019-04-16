
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
        TaskParameters.cpp
        TaskParameters.h
        Modules/Fiber/FiberProperties.cpp
        Modules/Fiber/FiberProperties.h
        Modules/Fiber/FiberSchedulerBackend.cpp
        Modules/Fiber/FiberSchedulerBackend.h
        Modules/Fiber/FiberSchedulerAlgorithm.cpp
        Modules/Fiber/FiberSchedulerAlgorithm.h
        Modules/Mock/MockSchedulerBackend.cpp
        Modules/Mock/MockSchedulerBackend.h
        SchedulerModule.cpp
        SchedulerModule.h
        Thread/ThreadLocal.cpp
        Thread/ThreadLocal.h
)
