#pragma once

#include "Algorithms/Morton Code/MortonCode.h"

inline int IndexLocal2D(int i, int j) {
	return MortonCode::Calculate64_2D(glm::uvec2(i,j));
}