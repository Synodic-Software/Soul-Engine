#include "TaggedHeap.h"

TaggedHeap::TaggedHeap(void* mem) {
	numAllocs = 0;
	usedSpace = 0;
	tags = new std::map<std::string, >();
	heap = mem;
}

void TaggedHeap::requestBlocks(std::string tag, size_t size) {
	// Tag does not exist, so create it within the heap
	if (tags.find(tag) == tags.end()) {
		
	}
}