#include"LinearAllocator.h"

void* LinearAllocator::allocate(size_t size, uint8_t alignment) {
	uint8_t alignment = alloc_tools::getAdjust(_currPos, alignment);
	if (_usedMem + alignment + size > _capacity) {
		return nullptr;
	}
	uint8_t alignedAddress = reinterpret_cast<uintptr_t>(_currPos) + alignment;
	_currPos += (void*) (alignedAddress + size);
	_usedMem += alignment + size;
	++_numAllocs;
	return (void*)alignedAddress;
}

LinearAllocator::~LinearAllocator() {
	_currPos = nullptr;
}

void* LinearAllocator::deallocate(void* block) {
	Assert.Fail("Linear Allocator does not support individual allocations.
		"Please use clear() instead");
}

void LinearAllocator::clear() {
	_numAllocs = 0;
	_usedMem = 0;
	_currPos = _start;
}