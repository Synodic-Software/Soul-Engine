#include "RasterJob.h"


//rasterjob
RasterJob::RasterJob() {

}

RasterJob::~RasterJob() {

}

RasterVariant const& RasterJob::operator [](std::string i) const {
	return shaderUniforms.at(i); 
}
RasterVariant& RasterJob::operator [](std::string i) {
	return shaderUniforms[i]; 
}
