#pragma once

#include "Ray Engine\CUDA\Ray.cuh"
#include <curand_kernel.h>
#include "Metrics.h"

class Ray;

//forward and right must be normalized!
    class Camera : public Managed {
    public:
		__host__ __device__ Camera();
		__host__ __device__ ~Camera();
         //The position of the camera.
        
        __host__ __device__ glm::vec3 Position() const;
		__host__ __device__ void SetPosition(const glm::vec3& position);
		__host__ __device__ void OffsetPosition(const glm::vec3& offset);

      
		__host__ __device__ glm::vec2 FieldOfView() const;
		__host__ __device__ void SetFieldOfView(glm::vec2 fieldOfView);

        /** A unit vector representing the direction the camera is facing */
		__host__ __device__ void SetForward(glm::vec3&);
		__host__ __device__ glm::vec3 Forward() const;

        /** A unit vector representing the direction to the right of the camera*/
		__host__ __device__ void SetRight(glm::vec3&);
		__host__ __device__ glm::vec3 Right() const;


		//Given a positive integer, this function fills in the given ray's values based on the camera's position orientation and lens.
	//	__host__ __device__ void SetupRay(uint&, Ray&, thrust::default_random_engine&, thrust::uniform_real_distribution<float>&);
		__device__ void SetupRay(uint&, Ray&, curandState&);

		__host__ __device__ bool IsViewable() const;

		__host__ __device__ void SetCircle(bool);

		__host__ __device__ void SetAspect(float);

		__host__ __device__ float GetAspect();

		__host__ __device__ void OffsetOrientation(float x, float y);

		__host__ __device__ void UpdateVariables();

		glm::uvec2 resolution;

    private:
		float aspectRatio;
		bool circularDistribution;
		
        glm::vec3 position;
		glm::vec3 forward;
		glm::vec3 right;
        glm::vec2 fieldOfView;
		float aperture;
		float focalDistance;

		//VARIABLE PRECALC
		glm::vec3 verticalAxis;
		glm::vec3 yHelper;
		glm::vec3 xHelper;
    };
