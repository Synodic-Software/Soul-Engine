#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <boost/filesystem.hpp>
#include <Frame.h>

namespace FrameManager {

	namespace detail {
		Frame currentframe;
	}

	void Step(Frame currentFrame) {
		currentFrame.frameCount += 1;
	}
}
