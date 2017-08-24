#pragma once

#include "Camera/Camera.h"
#include "GPGPU/GPUBuffer.h"

namespace CameraManager {

	namespace detail {
	}
	
	/* Initializes this object. */
	void Initialize();

	/* Terminates this object. */
	void Terminate();

	/* Updates this object. */
	void Update();

	/*
	 *    Adds a camera.
	 *    @param [in,out]	parameter1	The first parameter.
	 *    @return	Null if it fails, else a pointer to a Camera.
	 */

	Camera* AddCamera(glm::uvec2&);

	GPUBuffer<Camera>* GetCameraBuffer();

	/* Removes the camera. */
	void RemoveCamera();
}
