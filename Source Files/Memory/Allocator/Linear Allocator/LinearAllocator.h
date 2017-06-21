#pragma once
#include "Memory/Allocator/Allocator.h"
#include<cassert>

/*This class extends the base allocator class and provides
  the allocation logic for the linear allocator*/
class LinearAllocator : public Allocator {
	/*Function declarations*/
	public:
		LinearAllocator(size_t size, void * start);
		~LinearAllocator();

		void* allocate(size_t size, uint8_t alignment);
		void* deallocate(void* block);
		void clear();

	private:
		/*The allocator does not need to be copied, so the copy constructor
		  and assignment operator should be private*/
		LinearAllocator(const LinearAllocator&);
		LinearAllocator& operator=(const LinearAllocator&);
		void* _currPos; //keep track of next free space
};