#pragma once


/* Buffer for gpu. */
/* Buffer for gpu. */
class GPUBuffer {

public:
	/* Default constructor. */
	/* Default constructor. */
	GPUBuffer();
	/* Destructor. */
	/* Destructor. */
	~GPUBuffer();

	/*
	 *    Gets the data.
	 *
	 *    @return	Null if it fails, else the data.
	 */

	void* GetData() {
		return data;
	}

protected:
	/* The data */
	/* The data */
	void* data;

private:
	
};