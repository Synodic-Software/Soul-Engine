#pragma once

#include <vector>
#include "Engine Core\Object\Object.h"

class MiniObject {

public:

	MiniObject(Object&);

	bool requestRemoval;
	bool isStatic;

	uint verticeAmount;
	uint faceAmount;
	uint tetAmount;

protected:

private:


};
