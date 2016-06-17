#include "Image.h"

//uses stb_image to try load files
#define STB_IMAGE_IMPLEMENTATION
//#define STBI_FAILURE_USERMSG
#include <stb_image.h>


Image::Image()
	:width(0)
	, height(0)
	, format(0)
	, pixels(NULL)
{

}

Image::~Image() {
	if (pixels != NULL) {
		delete[] pixels;
	}
}

void Image::LoadFromFile(const char* filepath) {
	pixels = stbi_load(filepath, &width, &height, &format, 0);
}