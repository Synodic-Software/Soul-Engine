//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\GPGPU\GPUBuffer.h.
//Declares the GPU buffer class.

#pragma once


//Buffer for gpu.
class GPUBuffer {

public:
	//Default constructor.
	GPUBuffer();
	//Destructor.
	~GPUBuffer();

	//---------------------------------------------------------------------------------------------------
	//Gets the data.
	//@return	Null if it fails, else the data.

	void* GetData() {
		return data;
	}

protected:
	//The data
	void* data;

private:
	
};