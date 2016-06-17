#pragma once
#include "Utility\CUDAIncludes.h"

class Image: public Managed{
public:
	int width;
	int height;
	int format;
	unsigned char* pixels;

	Image();
	void LoadFromFile(const char*);
	~Image();
private:

};