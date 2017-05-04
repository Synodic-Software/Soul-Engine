#include "SceneNode.cuh"
#include "Algorithms\Morton Code\MortonCode.h"
#include <intrin.h>  

SceneNode::SceneNode(glm::mat4 tr) {

	uint64 mortonMin = MortonCode::CalculateMorton((tr*glm::vec4(-1.0f, -1.0f, -1.0f, 1.0f) + glm::vec4(1.0f)) / 2.0f);
	uint64 mortonMax = MortonCode::CalculateMorton((tr*glm::vec4(1.0f, 1.0f, 1.0f, 1.0f) + glm::vec4(1.0f)) / 2.0f);

	//auto isNonZero = _BitScanForward64(&bitsUsed,mortonMin^mortonMax);

	uint leadingZeros = _lzcnt_u64(mortonMin^mortonMax);
	morton = mortonMin >> (64- leadingZeros);

}
