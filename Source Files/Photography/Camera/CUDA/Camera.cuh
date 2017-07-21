#pragma once

#include "Ray Engine\CUDA\Ray.cuh"
#include "Photography/Film/Film.h"
#include <curand_kernel.h>

#include "Metrics.h"

class Camera {
public:
	Camera();
	~Camera();

	//Given a positive integer, this function fills in the given ray's values based on the camera's position orientation and lens.
	//__device__ void SetupRay(const uint&, Ray&, thrust::default_random_engine&, thrust::uniform_real_distribution<float>&);
	__device__ void GenerateRay(const uint, glm::vec3&, glm::vec3&, curandState&);

	void OffsetOrientation(float x, float y);

	void UpdateVariables();

	bool operator==(const Camera& other) const {
		return
			aspectRatio == other.aspectRatio &&
			position == other.position &&
			forward == other.forward &&
			right == other.right &&
			fieldOfView == other.fieldOfView &&
			aperture == other.aperture &&
			focalDistance == other.focalDistance &&
			verticalAxis == other.verticalAxis &&
			yHelper == other.yHelper &&
			xHelper == other.xHelper;
	}

	Camera& operator=(Camera arg)
	{
		this->aspectRatio = arg.aspectRatio;
		this->position = arg.position;
		this->forward = arg.forward;
		this->right = arg.right;
		this->fieldOfView = arg.fieldOfView;
		this->aperture = arg.aperture;
		this->focalDistance = arg.focalDistance;
		this->verticalAxis = arg.verticalAxis;
		this->yHelper = arg.yHelper;
		this->xHelper = arg.xHelper;

		return *this;
	}

	float aspectRatio;
	glm::vec3 position;
	glm::vec3 forward;
	glm::vec3 right;
	glm::vec2 fieldOfView;
	Film film;

private:
	float aperture;
	float focalDistance;

	//VARIABLE PRECALC
	glm::vec3 verticalAxis;
	glm::vec3 yHelper;
	glm::vec3 xHelper;

};
