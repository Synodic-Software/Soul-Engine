#include"LinearAllocator.h"

void LinearAllocator::allocate(size_t size, size_t alignment) {/*TODO implement allocations*/}

LinearAllocator::~LinearAllocator() {
	_currPos = nullptr;
}

void LinearAllocator::deallocate(void* block) {
	Assert.Fail("Linear Allocator does not support individual allocations.
		"Please use clear() instead");
}

void LinearAllocator::clear() {
	_numAllocs = 0;
	_usedMem = 0;
	_currPos = _start;
}