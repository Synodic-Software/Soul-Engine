#pragma once

#include "Types.h"
#include "Core/Composition/Component/Component.h"
#include "Core/Structure/External/ExternalVector.h"
#include "Core/Structure/External/ExternalArray.h"
#include "Core/Geometry/Vertex.h"

class RenderResource : public Component {

public:

	RenderResource() = default;
	~RenderResource() = default;

};

class RenderView : public RenderResource {

public:

	float width;
	float height;
	float minDepth;
	float maxDepth;

};

class Buffer : public RenderResource {

public:

	uint size;

};

class VertexBuffer : public Buffer{
};

class IndexBuffer : public Buffer {
};

class UniformBuffer : Buffer {
};

class PushBuffer : Buffer {
};

class StorageBuffer : Buffer {
};