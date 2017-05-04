#include "SceneNode.cuh"
#include "Algorithms\Morton Code\MortonCode.h"
#include <intrin.h>  

SceneNode::SceneNode(glm::mat4 tr) {

	uint64 mortonMin = MortonCode::CalculateMorton(tr*glm::vec4(-1.0f, -1.0f, -1.0f, 1.0f));
	uint64 mortonMax = MortonCode::CalculateMorton(tr*glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));

	uint64 leadingZeros = __lzcnt64(mortonMin);
	uint64 lowestBit = (64 - leadingZeros) - (__lzcnt64(mortonMin^mortonMax) - leadingZeros);
	morton = mortonMin >> lowestBit;

}
