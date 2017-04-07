#pragma once

#include <cuda_runtime.h>

class Image{
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