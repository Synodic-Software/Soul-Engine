#pragma once


#include "Compute\DeviceRasterBuffer.h"
#include "Compute\CUDA\CUDABuffer.h"

#include "Compute\ComputeDevice.h"


template <class T>
class CUDARasterBuffer : public CUDABuffer<T>, public DeviceRasterBuffer<T> {

public:

	//Types

	typedef T                                     value_type;
	typedef uint		                          size_type;


	//Construction and Destruction 

	CUDARasterBuffer(const ComputeDevice&);

	~CUDARasterBuffer();


	//Data Migration

	void MapResources()				override;
	void UnmapResources()			override;
	void BindData(uint)				override;
	void Resize(uint)				override;
	void Move(const ComputeDevice&) override;

private:


};

template <class T>
CUDARasterBuffer<T>::CUDARasterBuffer(const ComputeDevice& device) :
	DeviceRasterBuffer(device),
	CUDABuffer(device),
	DeviceBuffer(device)
{

}

template <class T>
CUDARasterBuffer<T>::~CUDARasterBuffer()
{

};

template <class T>
void CUDARasterBuffer<T>::MapResources()
{

}

template <class T>
void CUDARasterBuffer<T>::UnmapResources() {

}

template <class T>
void CUDARasterBuffer<T>::BindData(uint pos)
{

}

template <class T>
void CUDARasterBuffer<T>::Resize(uint newSize)
{
	if (newSize > 0) {

	}
}

template <class T>
void CUDARasterBuffer<T>::Move(const ComputeDevice&)
{

}