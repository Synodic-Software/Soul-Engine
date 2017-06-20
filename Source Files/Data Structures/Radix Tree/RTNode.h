//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\Data Structures\Radix Tree\RTNode.h.
//Declares the right node class.

#pragma once
#include <boost/dynamic_bitset.hpp>
#include "Metrics.h"
#include <vector>

//A right node.
class RTNode
{
public:
	//Default constructor.
	RTNode();

	//---------------------------------------------------------------------------------------------------
	//Constructor.
	//@param [in,out]	parameter1	The first parameter.

	RTNode(boost::dynamic_bitset<>&);

	//---------------------------------------------------------------------------------------------------
	//Adds a sequence.
	//@param [in,out]	parameter1	The first parameter.

	void AddSequence(boost::dynamic_bitset<>&);

private:
	//The bits
	boost::dynamic_bitset<> bits;
	//The children
	std::vector<RTNode> children;
};

