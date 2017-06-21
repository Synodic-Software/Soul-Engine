#pragma once
#include "GPGPU\GPUDevice.h"

/* An open cl device. */
/* An open cl device. */
class OpenCLDevice :public GPUDevice {

public:

	/*
	 *    Constructor.
	 *
	 *    @param	parameter1	The first parameter.
	 */

	OpenCLDevice(uint);
	/* Destructor. */
	/* Destructor. */
	virtual ~OpenCLDevice();

protected:

private:

};