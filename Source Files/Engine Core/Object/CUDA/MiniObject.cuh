#pragma once

#include <vector>
#include "Engine Core\Object\Object.h"
#include "Engine Core\Scene\SceneNode.h"

class MiniObject {

public:

	MiniObject(Object&);
	MiniObject();

	bool requestRemoval;
	bool isStatic;

	uint verticeAmount;
	uint faceAmount;
	uint tetAmount;

	SceneNode* transforms;
	uint tSize;
protected:

private:


};
