#pragma once
#include <boost/dynamic_bitset.hpp>
#include "Metrics.h"
#include <vector>

class RTNode
{
public:
	RTNode(boost::dynamic_bitset<>&);
	void AddSequence(boost::dynamic_bitset<>&);

private:
	boost::dynamic_bitset<> bits;
	std::vector<RTNode> children;
};

