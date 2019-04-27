#include "RenderGraphModule.h"

#include "Display/Window/WindowModule.h"
#include "Render/RenderGraph/Modules/Vulkan/VulkanRenderGraphBackend.h"


//TODO: This needs to instead be runtime loadable from shared libraries or statically linked
std::shared_ptr<RenderGraphModule> RenderGraphModule::CreateModule()
{

	return std::make_unique<VulkanRenderGraphBackend>();

}
