#pragma once
#include <map>

/* This class allows for used memory to be tagged on a per-module basis*/
class TaggedHeap {
	private:
		/*Private variables*/
		size_t numAllocs;
		void* heap;
		std::map<string, void*> tags; // associate module tag with its share of memory
		size_t usedSpace;
		
		/*Private functions*/
		void* addMem(std::string tag); //add additional memory to a tagged block
	
	public:
		TaggedHeap(void*); // Constructor for creating TaggedHeap
		void requestBlocks(std::string tag, size_t numBlocks, uint8_t align); // Request blocks from a specific tag
		void clearTag(std::string tag); // Clear memory from a given tag
		void clearAll(); // Clear the entire heap
};