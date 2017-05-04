#include "SceneNode.cuh"
#include "Algorithms\Morton Code\MortonCode.h"

SceneNode::SceneNode(int sc, glm::mat4 tr) {

	scale = sc;
	mortonMin = MortonCode::CalculateMorton(tr*glm::vec4(-1.0f,-1.0f,-1.0f,1.0f));
	mortonMax = MortonCode::CalculateMorton(tr*glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));

}
