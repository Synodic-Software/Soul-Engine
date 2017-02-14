#include "RasterJob.h"

//rasterjob
RasterJob::RasterJob(const std::vector<Shader>& shaders) {
	//detail::raster.get();
}

RasterJob::~RasterJob() {

}

RasterVariant const& RasterJob::operator [](std::string i) const { return shaderUniforms.at(i); }
RasterVariant& RasterJob::operator [](std::string i) { return shaderUniforms[i]; }