#include "RasterJob.h"


//rasterjob
RasterJob::RasterJob() {

}

RasterJob::~RasterJob() {

}

int const& RasterJob::operator [](std::string i) const {
	return shaderUniforms.at(i); 
}
int& RasterJob::operator [](std::string i) {
	return shaderUniforms[i]; 
}
