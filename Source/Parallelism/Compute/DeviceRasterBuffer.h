//Pure virtual function for the basis of all GPUBuffers

#pragma once

#include "Parallelism/Compute/ComputeDevice.h"

#include "DeviceBuffer.h"
/*
*    Buffer for gpu/cpu communication and storage.
*    @tparam	T	Generic type parameter.
*/

template <class T>
class DeviceRasterBuffer : public virtual  DeviceBuffer<T> {

public:


	DeviceRasterBuffer(const ComputeDevice& deviceIn);

	virtual ~DeviceRasterBuffer();


	virtual void MapResources() = 0;

	virtual void UnmapResources() = 0;

	virtual void BindData(uint) = 0;
	

};

template <class T>
DeviceRasterBuffer<T>::DeviceRasterBuffer(const ComputeDevice& deviceIn):
	DeviceBuffer(deviceIn) 
{

}

template <class T>
DeviceRasterBuffer<T>::~DeviceRasterBuffer()
{
	
}