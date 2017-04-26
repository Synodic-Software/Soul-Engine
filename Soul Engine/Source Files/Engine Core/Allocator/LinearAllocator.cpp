#include"LinearAllocator.h"

LinearAllocator::LinearAllocator(size_t size, void* start) : Allocator(size, start) {
	_currPos = start;
}

void* LinearAllocator::allocate(size_t size, uint8_t alignment) {
	uint8_t adjustment = alloc_tools::getAdjust(_currPos, alignment);
	if (_usedMem + adjustment + size > _capacity) {
		return nullptr;
	}
	uintptr_t* alignedAddress = (uintptr_t*)_currPos + adjustment;
	_currPos = (void*) (alignedAddress + size);
	_usedMem += adjustment + size;
	++_numAllocs;
	return (void*)alignedAddress;
}

LinearAllocator::~LinearAllocator() {
	_currPos = nullptr;
}

void* LinearAllocator::deallocate(void* block) {
    //deallocate should not be used for LinearAllocator
	return nullptr;
}

void LinearAllocator::clear() {
	_numAllocs = 0;
	_usedMem = 0;
	_currPos = _start;
}