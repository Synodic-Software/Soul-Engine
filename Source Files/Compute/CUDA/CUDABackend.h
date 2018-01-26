#pragma once

#include <vector>
#include "Compute/GPUBackendBase.h"

/* . */
class CUDABackend : public GPUBackendBase {

public:

	/*
	 *    Extracts the devices described by parameter1.
	 *    @param [in,out]	parameter1	The first parameter.
	 */

	void ExtractDevices(std::vector<CUDADevice>&);

	/* Initializes the thread. */
	void InitThread();

	/* Terminates this object. */
	void Terminate();


private:

};