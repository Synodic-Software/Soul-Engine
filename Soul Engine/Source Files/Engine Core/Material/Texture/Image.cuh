#pragma once
#include "Utility\CUDAIncludes.h"

class Image: public Managed{
public:
	int width;
	int height;
	int format;
	unsigned char* pixels;

	cudaTextureObject_t texObj;


	Image();
	void LoadFromFile(const char*, bool, bool);
	~Image();
private:

};