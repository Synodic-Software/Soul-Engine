
##############################################
#Composition

target_sources(${PROJECT_NAME}
    PRIVATE
        Component/Component.h
        Entity/Entity.cpp
        Entity/Entity.h
        Entity/EntityModule.cpp
        Entity/EntityModule.h
        Entity/Modules/Default/DefaultEntityBackend.cpp
        Entity/Modules/Default/DefaultEntityBackend.h
        Entity/Modules/Mock/MockEntityBackend.cpp
        Entity/Modules/Mock/MockEntityBackend.h
        Event/EventModule.cpp
        Event/EventModule.h
        Event/Modules/Default/DefaultEventBackend.cpp
        Event/Modules/Default/DefaultEventBackend.h
        Event/Modules/Default/Event.h
        Event/Modules/Mock/MockEventBackend.cpp
        Event/Modules/Mock/MockEventBackend.h
)
