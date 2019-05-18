#pragma once

#include "Types.h"
#include "Core/Composition/Component/Component.h"
class RenderResource : Component{

public:

	RenderResource() = default;
	~RenderResource() = default;

};

struct Buffer : RenderResource {

public:

	uint size;

};

struct VertexBuffer : Buffer {

};

struct IndexBuffer : Buffer {

};