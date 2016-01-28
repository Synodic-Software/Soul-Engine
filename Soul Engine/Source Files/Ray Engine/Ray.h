#pragma once

#   if defined(__CUDACC__)

#include "CUDA/Ray.cuh"

#	else

#include "OpenCL\CLRay.h"

#endif