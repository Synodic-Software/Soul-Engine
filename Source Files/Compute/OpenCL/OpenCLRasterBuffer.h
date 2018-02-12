#pragma once
#include "Compute\DeviceRasterBuffer.h"
#include "OpenCLBuffer.h"
#include "Compute\ComputeDevice.h"

#include "Metrics.h"

/* Buffer for open cl raster. */
template<class T>
class OpenCLRasterBuffer :public OpenCLBuffer<T>, public DeviceRasterBuffer<T> {

public:

	OpenCLRasterBuffer(const ComputeDevice& _device);

	~OpenCLRasterBuffer() override;


	void MapResources()override;

	void UnmapResources()override;

	void BindData(uint) override;

protected:

private:

};

template<class T>
OpenCLRasterBuffer<T>::OpenCLRasterBuffer(const ComputeDevice& device) :
	OpenCLBuffer(device),
	DeviceRasterBuffer(device),
	DeviceBuffer(device)
{

}

template<class T>
OpenCLRasterBuffer<T>::~OpenCLRasterBuffer()
{
	
}

template<class T>
void OpenCLRasterBuffer<T>::MapResources()
{
	
}

template<class T>
void OpenCLRasterBuffer<T>::UnmapResources()
{
	
}

template<class T>
void OpenCLRasterBuffer<T>::BindData(uint)
{
	
}