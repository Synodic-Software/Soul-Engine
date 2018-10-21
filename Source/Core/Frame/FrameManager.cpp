#include "FrameManager.h"

const Frame& FrameManager::Next() {

	frames_.Push(Frame());

	return frames_[2];

}