#include "Sort.cuh"

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

namespace Sort {

	void Sort(GPUBuffer<uint64_t>& keys, GPUBuffer<Face>& values) {

		thrust::device_ptr<uint64_t> keysP(keys.DeviceData());
		thrust::device_ptr<Face> valuesP(values.DeviceData());

		sort_by_key(keysP, keysP + keys.DeviceSize(), valuesP);

	}

}
