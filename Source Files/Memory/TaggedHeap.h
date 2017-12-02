#pragma once
#include <map>
#include <list>
/* This class allows for used memory to be tagged on a per-module basis*/
class TaggedHeap {
	private:
		/*Private variables*/
		size_t numAllocs;
		void* heap;
		std::map<std::string, std::list<std::pair<void*, size_t>>> tags; // associate module tag with its share of memory
		size_t usedSpace;

	
	public:
		TaggedHeap(void*); // Constructor for creating TaggedHeap
		void requestBlocks(std::string tag, size_t size); // Request blocks from a specific tag
		void clearTag(std::string tag); // Clear memory from a given tag
		void clearAll(); // Clear the entire heap
};