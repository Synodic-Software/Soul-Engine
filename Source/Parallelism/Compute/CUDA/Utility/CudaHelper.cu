#include "CudaHelper.cuh"

__host__ __device__ uint randHash(uint a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}


__device__ int warp_bcast(int v, int leader) { return __shfl(v, leader); }
__device__ int lane_id() { return threadIdx.x % WARP_SIZE; }

// warp-aggregated atomic increment
__device__
int FastAtomicAdd(int *ctr) {
	int mask = __ballot(1);
	// select the leader
	int leader = __ffs(mask) - 1;
	// leader does the update
	int res;
	if (lane_id() == leader)
		res = atomicAdd(ctr, __popc(mask));
	// broadcast result
	res = warp_bcast(res, leader);
	// each thread computes its own value
	return res + __popc(mask & ((1 << lane_id()) - 1));
} // FastAtomicAdd