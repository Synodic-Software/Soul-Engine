#pragma once

#include "Camera/Camera.h"

namespace CameraManager {

	namespace detail {
	}

	/* Updates this object. */
	void Update();

	/*
	 *    Adds a camera.
	 *    @param [in,out]	parameter1	The first parameter.
	 *    @return	Null if it fails, else a pointer to a Camera.
	 */

	Camera* AddCamera(glm::uvec2&);

	/* Removes the camera. */
	void RemoveCamera();
}