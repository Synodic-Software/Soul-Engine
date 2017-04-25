#pragma once
#include "Allocator.h"

class LinearAllocator : public Allocator {
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
	void* _currPos;
};