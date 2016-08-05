#pragma once
#include "Utility\CUDAIncludes.h"

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