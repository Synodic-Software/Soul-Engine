#pragma once

#include "Utility\CUDAIncludes.h"
#include "Ray Engine\CUDA/Ray.cuh"
#include "thrust\random.h"


//forward and right must be normalized!
    class Camera : public Managed {
    public:
		CUDA_FUNCTION Camera();
		CUDA_FUNCTION ~Camera();
         //The position of the camera.
        
        CUDA_FUNCTION glm::vec3 Position() const;
		CUDA_FUNCTION void SetPosition(const glm::vec3& position);
		CUDA_FUNCTION void OffsetPosition(const glm::vec3& offset);

      
		CUDA_FUNCTION glm::vec2 FieldOfView() const;
		CUDA_FUNCTION void SetFieldOfView(glm::vec2 fieldOfView);

        /** A unit vector representing the direction the camera is facing */
		CUDA_FUNCTION void SetForward(glm::vec3&);
		CUDA_FUNCTION glm::vec3 Forward() const;

        /** A unit vector representing the direction to the right of the camera*/
		CUDA_FUNCTION void SetRight(glm::vec3&);
		CUDA_FUNCTION glm::vec3 Right() const;


		//Given a positive integer, this function fills in the given ray's values based on the camera's position orientation and lens.
		CUDA_FUNCTION void SetupRay(uint&, Ray&, thrust::default_random_engine&, thrust::uniform_real_distribution<float>&);

		CUDA_FUNCTION bool IsViewable() const;

		CUDA_FUNCTION void SetCircle(bool);

		CUDA_FUNCTION void SetAspect(float);

		CUDA_FUNCTION float GetAspect();

		CUDA_FUNCTION void OffsetOrientation(float x, float y);

		CUDA_FUNCTION void UpdateVariables();

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
