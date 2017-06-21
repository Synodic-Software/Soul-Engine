#pragma once

#include "RTNode.h"
#include <vector>
#include "Metrics.h"
//#include <boost/multiprecision/gmp.hpp>

//namespace boost {
//	namespace multiprecision {
//		class gmp_int;
//		typedef number<gmp_int> mpInt;
//	}
//}
//
//typedef boost::multiprecision::mpInt ArbitraryInt;


/* A radix tree. */
/* A radix tree. */
class RadixTree
{
public:

	/*
	 *    Constructor.
	 *
	 *    @param	parameter1	The first parameter.
	 */

	RadixTree(uint);

	/*
	 *    Adds a sequence.
	 *
	 *    @param [in,out]	parameter1	The first parameter.
	 */

	void AddSequence(boost::dynamic_bitset<>&);

private:
	/* The root */
	/* The root */
	RTNode root;
	/* The bits to split */
	/* The bits to split */
	uint bitsToSplit;
};

