#pragma once

#include "Types.h"
#include "Core/Composition/Component/Component.h"
#include "Core/Geometry/Vertex.h"

enum class ResourceType { 
	Buffer, 
	Image
};

class RenderResource : public Component {

public:

	RenderResource(ResourceType typeIn): type(typeIn)
	{
	}
	~RenderResource() = default;

	ResourceType type;

};

//TODO: Place somewhere else
class RenderView : public Component {

public:

	float width;
	float height;
	float minDepth;
	float maxDepth;

};



enum class BufferType {
	Vertex,
	Index,
	Uniform,
	Push,
	Storage
};

class Buffer : public RenderResource {

public:

	Buffer(BufferType typeIn): 
		RenderResource(ResourceType::Buffer),
		type(typeIn)
	{
	}
	~Buffer() = default;


private:

	BufferType type;


};

class VertexBuffer : public Buffer{

public:

	VertexBuffer(): Buffer(BufferType::Vertex)
	{
	}
	~VertexBuffer() = default;

};

class IndexBuffer : public Buffer {

public:

	IndexBuffer(): Buffer(BufferType::Index)
	{
	}
	~IndexBuffer() = default;

};

class UniformBuffer : public Buffer {

public:

	UniformBuffer(): Buffer(BufferType::Uniform)
	{
	}
	~UniformBuffer() = default;

};

class PushBuffer : public Buffer {

public:

	PushBuffer() : Buffer(BufferType::Push)
	{
	}
	~PushBuffer() = default;

};

class StorageBuffer : public Buffer {


public:

	StorageBuffer(): Buffer(BufferType::Storage)
	{
	}
	~StorageBuffer() = default;

};