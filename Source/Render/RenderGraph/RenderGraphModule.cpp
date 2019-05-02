#include "RenderGraphModule.h"

#include "Display/Window/WindowModule.h"
#include "Render/RenderGraph/Modules/Entity/EntityRenderGraphBackend.h"
#include "Render/RenderGraph/RenderGraphModule.h"


//TODO: This needs to instead be runtime loadable from shared libraries or statically linked
std::shared_ptr<RenderGraphModule> RenderGraphModule::CreateModule(std::shared_ptr<RasterModule>& rasterModule)
{

	return std::make_unique<EntityRenderGraphBackend>(rasterModule);

}
