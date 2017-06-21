#pragma once
#include <boost/dynamic_bitset.hpp>
#include "Metrics.h"
#include <vector>

/* A right node. */
/* A right node. */
class RTNode
{
public:
	/* Default constructor. */
	/* Default constructor. */
	RTNode();

	/*
	 *    Constructor.
	 *
	 *    @param [in,out]	parameter1	The first parameter.
	 */

	RTNode(boost::dynamic_bitset<>&);

	/*
	 *    Adds a sequence.
	 *
	 *    @param [in,out]	parameter1	The first parameter.
	 */

	void AddSequence(boost::dynamic_bitset<>&);

private:
	/* The bits */
	/* The bits */
	boost::dynamic_bitset<> bits;
	/* The children */
	/* The children */
	std::vector<RTNode> children;
};

