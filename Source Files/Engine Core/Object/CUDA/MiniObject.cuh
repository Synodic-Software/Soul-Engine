#pragma once

#include <vector>
#include "Engine Core\Object\Object.h"

class MiniObject {

public:

	MiniObject(Object&);
	MiniObject();

	bool requestRemoval;
	bool isStatic;

	uint verticeAmount;
	uint faceAmount;
	uint tetAmount;

	uint offSet;

protected:

private:


};
