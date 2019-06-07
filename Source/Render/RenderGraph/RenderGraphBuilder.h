#pragma once

#include "RenderGraphParameters.h"
#include "Render/Raster/RenderResource.h"
#include "Core/Composition/Entity/EntityRegistry.h"
#include "Render/Raster/RasterModule.h"

class RenderGraphBuilder{

public:

	RenderGraphBuilder(std::shared_ptr<RasterModule>&, std::shared_ptr<EntityRegistry>&);
	virtual ~RenderGraphBuilder() = default;

	RenderGraphBuilder(const RenderGraphBuilder &) = delete;
	RenderGraphBuilder(RenderGraphBuilder &&) noexcept = default;

	RenderGraphBuilder& operator=(const RenderGraphBuilder &) = delete;
	RenderGraphBuilder& operator=(RenderGraphBuilder &&) noexcept = default;

	void CreateOutput(RenderGraphOutputParameters&);
	void CreateInput(RenderGraphInputParameters&);

	void CreateSubpass();

	template<class T>
	Entity Request();

	Entity View();

private:

	std::shared_ptr<EntityRegistry> entityRegistry_;
	std::shared_ptr<RasterModule> rasterModule_;

};

template<class T>
Entity RenderGraphBuilder::Request()
{
	// TODO: C++20 Concepts
	static_assert(std::is_base_of<RenderResource, T>::value,
		"The type parameter must be a subclass of RenderResource"); 


	Entity returnedEntity = entityRegistry_->CreateEntity();
	entityRegistry_->AttachComponent<T>(returnedEntity);

	return returnedEntity;

}
