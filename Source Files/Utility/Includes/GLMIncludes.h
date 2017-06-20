//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\Utility\Includes\GLMIncludes.h.
//Declares the glm includes class.

#pragma once

//#  if defined(__CUDACC__)
#include <cuda.h>  

#define GLM_FORCE_CUDA
//#endif
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtc/noise.hpp>
//#include <glm/gtx/rotate_vector.hpp>
