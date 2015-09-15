#include "Vertex.h"


Vertex::Vertex()
{
	position = new float3(make_float3(0.0f, 0.0f, 0.0f));
	textureCoord = new float2(make_float2(0.0f, 0.0f));
	normal = new float3(make_float3(0.0f, 0.0f, 0.0f));
}
Vertex::Vertex(float3 posTemp, float2 uvTemp, float3 normTemp)
{
	position = new float3(posTemp);
	textureCoord = new float2(uvTemp);
	normal = new float3(normTemp);
}

Vertex::~Vertex()
{
	delete position;
	delete textureCoord;
	delete normal;
}
