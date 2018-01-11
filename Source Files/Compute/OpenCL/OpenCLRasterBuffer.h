#pragma once
#include "Compute\DeviceRasterBuffer.h"
#include "Compute\GPUDevice.h"

#include "Metrics.h"

/* Buffer for open cl raster. */
template<class T>
class OpenCLRasterBuffer :public DeviceRasterBuffer<T> {

public:

	/*
	 *    Constructor.
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *    @param 		 	parameter2	The second parameter.
	 */

	OpenCLRasterBuffer(GPUDevice& _device, uint _byteCount)
		: DeviceRasterBuffer(_device, _byteCount) {
		
	}
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