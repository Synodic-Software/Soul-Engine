#pragma once

#include "Metrics.h"
#include "Compute/ComputeBuffer.h"

class BitList
{

public:

	BitList();


private:

	ComputeBuffer<uint> sizes;
	ComputeBuffer<uint> offsets;
	ComputeBuffer<char> data;

};

