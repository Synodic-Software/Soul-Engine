
##############################################
#Composition

target_sources(${PROJECT_NAME}
    PRIVATE
        Component/Component.h
        Entity/Entity.cpp
        Entity/Entity.h
        Entity/EntityManager.cpp
        Entity/EntityManager.cpp
        Entity/SparseEntitySet/SparseEntitySet.h
        Entity/SparseEntitySet/SparseEntitySet.cpp
        Entity/SparseEntitySet/AbstractSparseEntitySet.h
        Entity/SparseEntitySet/AbstractSparseEntitySet.cpp
        Event/Event.h
        Event/EventManager.h
        Event/EventManager.cpp
)
