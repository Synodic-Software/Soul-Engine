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
};

class VertexBuffer final : public  Buffer, ExternalVector<RenderVertex>{

public:

};

template<class T>
class IndexBuffer final : public Buffer {

	static_assert(std::is_integral<float>::value, "Type T must be an integral type.");

public:

};

template<class T>
class UniformBuffer final : Buffer {

public:

};

template<class T>
class PushBuffer final : Buffer {

public:

};

template<class T>
class StorageBuffer final : Buffer {

public:
};