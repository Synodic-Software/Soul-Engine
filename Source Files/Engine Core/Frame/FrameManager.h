#pragma once

#include "Frame.h"

namespace FrameManager {

	namespace detail {
		Frame currentframe;
	}

	void Step(Frame currentFrame) {
		currentFrame.frameCount += 1;
	}
}
