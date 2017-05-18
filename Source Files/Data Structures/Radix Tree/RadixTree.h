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


class RadixTree
{
public:


	RadixTree(uint);

	void AddSequence(boost::dynamic_bitset<>&);

private:
	RTNode root;
	uint bitsToSplit;
};

