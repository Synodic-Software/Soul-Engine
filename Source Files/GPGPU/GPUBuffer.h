#pragma once


/* Buffer for gpu. */

template <class T>
class GPUBuffer {

public:
	/* Default constructor. */
	GPUBuffer(){}
	/* Destructor. */
	virtual ~GPUBuffer(){}

	/*
	 *    Gets the data.
	 *    @return	Null if it fails, else the data.
	 */

	void* Data() {
		return data;
	}

protected:
	/* The data */
	void* data;

private:
	
};