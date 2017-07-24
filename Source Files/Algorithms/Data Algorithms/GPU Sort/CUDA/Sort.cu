//#include "Sort.cuh"

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "Utility/CUDA/CudaHelper.cuh"
#include "Metrics.h"

namespace Sort {

	uint* Calculate(uint size, uint64* keys, uint* values) {

		uint64* keysDevice;
		uint* valuesDevice;

		//create device space
		CudaCheck(cudaMalloc((void **)&keysDevice, size * sizeof(uint64)));
		CudaCheck(cudaMalloc((void **)&valuesDevice, size * sizeof(uint)));

		//tranfer memory
		CudaCheck(cudaMemcpy(keysDevice, keys, size * sizeof(uint64), cudaMemcpyHostToDevice));
		CudaCheck(cudaMemcpy(valuesDevice, values, size * sizeof(uint), cudaMemcpyHostToDevice));

		//sort with thrust
		thrust::device_ptr<uint64> key_ptr = thrust::device_pointer_cast(keysDevice);
		thrust::device_ptr<uint> value_ptr = thrust::device_pointer_cast(valuesDevice);

		thrust::sort_by_key(key_ptr, key_ptr + size, value_ptr);

		CudaCheck(cudaFree(keysDevice));

		return valuesDevice;
	}

}