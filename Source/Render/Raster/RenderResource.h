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

class VertexBuffer : public  Buffer{

public:

	nonstd::span<RenderVertex> vertices;

};

//TODO: variable index sizes
class IndexBuffer : public Buffer {

public:

	nonstd::span<uint16> indices;

};

template<class T>
class UniformBuffer : Buffer{
public:
};


//TODO: make constant
//N = Number of bytes
template<class T>
class PushConstant : Buffer {

public:

	PushConstant() = default;
	~PushConstant() = default;

	T* pushConstant;

};