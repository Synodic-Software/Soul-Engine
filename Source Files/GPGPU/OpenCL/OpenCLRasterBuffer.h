#pragma once
#include "GPGPU\GPURasterBuffer.h"
#include "GPGPU\OpenCL\OpenCLDevice.h"

#include "Metrics.h"

/* Buffer for open cl raster. */
template<class T>
class OpenCLRasterBuffer :public GPURasterBuffer<T> {

public:

	/*
	 *    Constructor.
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *    @param 		 	parameter2	The second parameter.
	 */

	OpenCLRasterBuffer(OpenCLDevice*, uint){}
	/* Destructor. */
	~OpenCLRasterBuffer(){}

	/* Map resources. */
	void MapResources(){}
	/* Unmap resources. */
	void UnmapResources(){}

	/*
	 *    Bind data.
	 *    @param	parameter1	The first parameter.
	 */

	void BindData(uint){}

protected:

private:

};