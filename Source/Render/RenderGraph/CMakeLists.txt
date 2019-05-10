
##############################################
#Rasterer

target_sources(${PROJECT_NAME}
    PRIVATE
        Modules/Entity/EntityRenderGraphBackend.cpp
        Modules/Entity/EntityRenderGraphBackend.h
        RenderTaskParameters.cpp
        RenderTaskParameters.h
        RenderGraphBuilder.cpp
        RenderGraphBuilder.h
        RenderGraphModule.cpp
        RenderGraphModule.h
)
