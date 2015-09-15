#pragma once

#include "Engine Core\BasicDependencies.h"
#include "Utility\CUDA\CUDAManaged.cu"
#include "Engine Core\Object\Vertex.h"
#include "Engine Core\Object\Face.h"

class ObjectProperties : public Managed{
public:
	ObjectProperties();
	void AddVertices(std::vector<Vertex>&);
	void AddFaces(std::vector<Face>&);
private:
	Vertex* vertices;
	Face* faces;

	float3* translate;
	float3* rotate;
	float3* scale;

	bool removeRequested;
};