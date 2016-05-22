#pragma once

#include "Utility\CUDAIncludes.h"
#include <iostream>
#include "cuda.h"

#define CUDA_FUNCTION __host__ __device__

#define CudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		std::cout<<cudaGetErrorString(code)<< file<< line<<std::endl;
		//if (abort) exit(code);
	}
}

CUDA_FUNCTION uint randHash(uint a);
inline __device__ int getGlobalIdx_1D_1D()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}

//template <class T> __device__ __inline__ void swap(T& a, T& b)
//{
//	T t = a;
//	a = b;
//	b = t;
//}
//
//__device__ __inline__ int __float_as_int(float in) {
//	union fi { int i; float f; } conv;
//	conv.f = in;
//	return conv.i;
//}
//
//__device__ __inline__ float __int_as_float(int a)
//
//{
//
//	union { int a; float b; } u;
//
//	u.a = a;
//
//	return u.b;
//
//}
//
//__device__ __inline__ float min4(float a, float b, float c, float d)
//{
//	return fminf(fminf(fminf(a, b), c), d);
//}
//
//__device__ __inline__ float max4(float a, float b, float c, float d)
//{
//	return fmaxf(fmaxf(fmaxf(a, b), c), d);
//}
//
//__device__ __inline__ float min3(float a, float b, float c)
//{
//	return fminf(fminf(a, b), c);
//}
//
//__device__ __inline__ float max3(float a, float b, float c)
//{
//	return fmaxf(fmaxf(a, b), c);
//}
//
//// Using integer min,max
//__inline__ __device__ float fminf2(float a, float b)
//{
//	int a2 = __float_as_int(a);
//	int b2 = __float_as_int(b);
//	return __int_as_float(a2<b2 ? a2 : b2);
//}
//
//__inline__ __device__ float fmaxf2(float a, float b)
//{
//	int a2 = __float_as_int(a);
//	int b2 = __float_as_int(b);
//	return __int_as_float(a2>b2 ? a2 : b2);
//}
//
//// Using video instructions
//__device__ __inline__ int   min_min(int a, int b, int c) { int v; asm ("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
//__device__ __inline__ int   min_max(int a, int b, int c) { int v; asm ("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
//__device__ __inline__ int   max_min(int a, int b, int c) { int v; asm ("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
//__device__ __inline__ int   max_max(int a, int b, int c) { int v; asm ("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
//__device__ __inline__ float fmin_fmin(float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
//__device__ __inline__ float fmin_fmax(float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
//__device__ __inline__ float fmax_fmin(float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
//__device__ __inline__ float fmax_fmax(float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
//
//
//__device__ __inline__ float magic_max7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
//{
//	float t1 = fmin_fmax(a0, a1, d);
//	float t2 = fmin_fmax(b0, b1, t1);
//	float t3 = fmin_fmax(c0, c1, t2);
//	return t3;
//}
//
//__device__ __inline__ float magic_min7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
//{
//	float t1 = fmax_fmin(a0, a1, d);
//	float t2 = fmax_fmin(b0, b1, t1);
//	float t3 = fmax_fmin(c0, c1, t2);
//	return t3;
//}
//
//// Span for Fermi.
//__device__ __inline__ float spanBegin(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return magic_max7(a0, a1, b0, b1, c0, c1, d); }
//__device__ __inline__ float spanEnd(float a0, float a1, float b0, float b1, float c0, float c1, float d)	{ return magic_min7(a0, a1, b0, b1, c0, c1, d); }
