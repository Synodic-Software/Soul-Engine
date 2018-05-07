#include "Sort.cuh"

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

namespace Sort {

	void Sort(ComputeBuffer<uint64_t>& keys, ComputeBuffer<Face>& values) {

		thrust::device_ptr<uint64_t> keysP(keys.DataDevice());
		thrust::device_ptr<Face> valuesP(values.DataDevice());

		sort_by_key(keysP, keysP + keys.SizeDevice(), valuesP);

	}

}
