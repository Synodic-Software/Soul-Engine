#pragma once
#include "LinearAllocator.h"
#include "TaggedHeap.h"
#include <list>

/*Tagged Allocator is an allocator class that utilizes the tagged heap for per-module memory management.
  It uses a LinearAllocator under the hood to make the actual allocations, but obtains memory from the TaggedHeap.
  Memory allocation happens on a per-thread bases to make this allocator thread-safe*/
class TaggedAllocator {
	private:
		// Private variables
		TaggedHeap heap;
		LinearAllocator allocator
		std::map<long, std::list<std::pair<void*, size_t>>> blocks;

	public:
		TaggedAllocator(TaggedHeap heap, std::string tag);
		void* allocate(long threadId, size_t, uint8_t);
		void* deallocate(void* block);

};