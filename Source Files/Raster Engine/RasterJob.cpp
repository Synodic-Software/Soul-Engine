//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\Raster Engine\RasterJob.cpp.
//Implements the raster job class.

#include "RasterJob.h"


//rasterjob.
RasterJob::RasterJob() {

}

//Destructor.
RasterJob::~RasterJob() {

}

//---------------------------------------------------------------------------------------------------
//Array indexer operator.
//@param	i	Zero-based index of the.
//@return	The indexed value.

int const& RasterJob::operator [](std::string i) const {
	return shaderUniforms.at(i); 
}

//---------------------------------------------------------------------------------------------------
//Array indexer operator.
//@param	i	Zero-based index of the.
//@return	The indexed value.

int& RasterJob::operator [](std::string i) {
	return shaderUniforms[i]; 
}
