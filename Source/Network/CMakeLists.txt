
##############################################
#Network

target_sources(${PROJECT_NAME}
    PRIVATE
        AbstractNetworkManager.cpp
        AbstractNetworkManager.h
        Client/Client.cpp
        Client/Client.h
        NetworkManager.cpp
        NetworkManager.h
        Server/Server.cpp
        Server/Server.h
)
