#pragma once

#include "Engine Core\BasicDependencies.h"

class Node : public Managed
{
public:
	Node();
	~Node();

private:
	float systemMin = 0.0f;
	float systemMax = 1.0f;


};