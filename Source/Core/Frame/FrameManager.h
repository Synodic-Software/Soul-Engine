#pragma once

#include "Frame.h"
#include "Core/Structures/RingBuffer.h"

class FrameManager {

public:

	FrameManager() = default;
	~FrameManager() = default;

	const Frame& Next();

private:

	RingBuffer<Frame, 3> frames_;

};