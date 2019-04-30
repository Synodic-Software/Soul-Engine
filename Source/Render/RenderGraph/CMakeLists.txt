
##############################################
#Rasterer

target_sources(${PROJECT_NAME}
    PRIVATE
        Modules/Vulkan/VulkanRenderGraphBackend.cpp
        Modules/Vulkan/VulkanRenderGraphBackend.h
        RenderGraphModule.cpp
        RenderGraphModule.h
)
