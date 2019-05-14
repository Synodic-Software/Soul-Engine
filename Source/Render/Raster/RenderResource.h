#pragma once

#include "Types.h"

class RenderResource{

public:

	RenderResource() = default;
	~RenderResource() = default;

	RenderResource(const RenderResource&) = default;
	RenderResource(RenderResource &&) noexcept = default;

	RenderResource& operator=(const RenderResource&) = default;
	RenderResource& operator=(RenderResource &&) noexcept = default;


};

struct Buffer : RenderResource {

};

struct VertexBuffer : Buffer {

};

struct IndexBuffer : Buffer {

};