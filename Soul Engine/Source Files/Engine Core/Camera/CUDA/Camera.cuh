#pragma once

#include "Engine Core\BasicDependencies.h"
#include "Ray Engine\Ray.cuh"

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

		CUDA_FUNCTION Ray SetupRay(uint, uint, thrust::default_random_engine);

		CUDA_FUNCTION bool IsViewable() const;

		CUDA_FUNCTION void SetCircle(bool cir);

		CUDA_FUNCTION void SetResolution(glm::uvec2 res);

		CUDA_FUNCTION glm::uvec2 GetResolution();

		CUDA_FUNCTION void OffsetOrientation(float x, float y);

    private:
		bool circularDistribution;
		glm::uvec2 resolution;
        glm::vec3 position;
		glm::vec3 forward;
		glm::vec3 right;
        glm::vec2 fieldOfView;
		float aperture;
		float focalDistance;
    };
