#pragma once
#include <cstdint>

/*This class will serve as the base class for the engine's linear allocator
  it contains public allocate and deallocate functions which will manage the memory
  These must be implemented by the child linear allocator class itself.*/
class Allocator {
	public:
		Allocator(size_t size, void* start);
		virtual ~Allocator();
		virtual void* allocate(size_t size, uint8_t alignment = 4) = 0;
		virtual void* deallocate(void* block) = 0;
		void* getStart() {return _start;}
		size_t getCapacity() {return _capacity;}
		size_t getUsedMem() {return _usedMem;}
		size_t getNumAllocs() {return _numAllocs;}

	protected:
		void* _start;
		size_t _capacity;
		size_t _usedMem;
		size_t _numAllocs;


};
/*Allocates space for a new object of type T using the specified allocator
 Arguments: allocator - a reference to the allocator to use
 Returns: a pointer to the newly allocated memory*/
template<Class T> T* Allocator::allocateNew(Allocator& allocator) {
	return new (allocator.allocate(sizeof(T), __alignof(T))) T;
}

/*Allocates space for object t using the specified allocator
Arguments: allocator - a reference to the allocator to use
		   t - a reference to the object to store in the newly allocated memory
Returns: a pointer to the newly allocated memory*/
template<Class T> T* Allocator::allocateNew(Allocator& allocator, const T& t) {
	return new (allocator.allocate(sizeof(T), __alignof(T))) T(t);
}

/*Allocates space for an array of the specified type and length
 Arguments: allocator - a reference to the allocator to use
	        length - the amount of elements the array will hold
 Returns: a pointer to the newly allocated array*/
template<Class T> T* Allocator::allocateNewArr(Allocator& allocator, size_t length) {
	size_t headerSize = sizeof(size_t)/sizeof(T);
	if (sizeof(size_t) % sizeof(T) > 0) {
		headerSize += 1;
	}
	T* arr = (T*) allocator.allocate(sizeof(T)*(length + headerSize), __alignof(T)) + headerSize;
	*(((size_t*)arr) - 1) = length;
	for (size_t i = 0; i < length; ++i) {
		*(arr + i) = new T();
	}
	return arr;
}

/*Deallocate the specified item
 Arguments: allocator - the allocator to use
			object - the object to deallocate
 Returns: none*/
template<Class T> void Allocator::deallocateItem(Allocator& allocator, T& object) {
	object.~T();
	allocator.deallocate(&object);
}

/*Deallocate the specified array
 Arguments: allocator - the allocator to use
		    arr - the array to deallocate
 Returns: none 
*/
template<Class T> void Allocator::deallocateArr(Allocator& allocator, T* arr) {
	size_t length = *(((size_t*)arr) - 1);
	for (size_t i = 0; i < length; ++i) {
		*(arr + i).~T();
	}
	size_t headerSize = sizeof(size_t) / sizeof(T);
	if (sizeof(size_t) % sizeof(T) > 0) {
		headerSize += 1;
	}
	allocator.deallocate(arr - headerSize);
}

/*Tools to help make aligned allocatoins*/
namespace alloc_tools {
	inline void* align(void* address, uint8_t alignment) {
		return (void*) ((reinterpret_cast<uintptr_t>(address) + alignment - 1) & ~(alignment - 1));
	}

	inline uint8_t getAdjust(void* address, uint8_t alignment, uint8_t headerSize = 0) {
		uint8_t adjustment = alignment - (reinterpret_cast<uintptr_t>(address) & (alignment - 1));
		if (adjustment == alignment) {
			adjustment = 0;
		}
		if (adjustment < headerSize) {
			headerSize -= adjustment;
			adjustment += alignment * (headerSize / alignment);
			if (headerSize % alignment > 0) {
				adjustment += alignment;
			}
		}
		return adjustment;
	}
}
