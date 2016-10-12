#pragma once

#include "Utility\CUDA\CUDAManaged.cuh"
#include <cuda_runtime.h>

class Image: public Managed{
public:
	int width;
	int height;
	int format;
	void* pixels;

	cudaTextureObject_t texObj;


	Image();
	void LoadFromFile(const char*, bool, bool);
	~Image();
private:

};

unsigned char* DirectLoad(const char*, int*, int*, int*);