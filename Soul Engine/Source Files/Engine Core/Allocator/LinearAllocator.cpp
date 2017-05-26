#include"LinearAllocator.h"

/*Constructor for linear allocator
  Arguments: size - maximum capacity in bytes for the allocator
			 start - the buffer used to make allocations
*/
LinearAllocator::LinearAllocator(size_t size, void* start) : Allocator(size, start) {
	_currPos = start;
}

/*Function used to allocate memory for the allocator
  Arguments: size - the amount of space to allocate
			 alignment - the alignment for the allocation
  Returns - pointer to allocated space*/
void* LinearAllocator::allocate(size_t size, uint8_t alignment) {
	/*Get amount of bytes needed to align the address*/
	uint8_t adjustment = alloc_tools::getAdjust(_currPos, alignment);
	
	/*Ensure that their is enough space in the buffer to allocate new space*/
	if (_usedMem + adjustment + size > _capacity) {
		return nullptr;
	}

	/*Calculate the align the address and update the allocator's state accordingly*/
	uintptr_t* alignedAddress = (uintptr_t*)_currPos + adjustment;
	_currPos = (void*) (alignedAddress + size);
	_usedMem += adjustment + size;
	++_numAllocs;
	return (void*)alignedAddress;
}

/*Destructor for allocator*/
LinearAllocator::~LinearAllocator() {
	_currPos = nullptr;
}

/*NOTE: this function is not actually utilized for the linear allocator, but is needed
  since the allocator base class has purely virtual deallocate function*/
void* LinearAllocator::deallocate(void* block) {
	assert(false);
	return nullptr;
}

/*Clear the allocator by resetting it to the default state*/
void LinearAllocator::clear() {
	_numAllocs = 0;
	_usedMem = 0;
	_currPos = _start;
}