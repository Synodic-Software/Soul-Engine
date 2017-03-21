#pragma once
#include<cassert>
/*This class will serve as the base class for the engine's linear allocator
  it contains public allocate and deallocate functions which will manage the memory
  These must be implemented by the child linear allocator class itself.*/
class Allocator {
	public:
		Allocator(size_t size, void* start);
		virtual ~Allocator();
		virtual void* allocate(size_t size, u8 alignment = 4) = 0;
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
template<Class T> T* Allocator::allocateNew(Allocator& allocator) {
	return new (allocator.allocate(sizeof(T), __alignof(T))) T;
}

template<Class T> T* Allocator::allocateNew(Allocator& allocator, const T& t) {
	return new (allocator.allocate(sizeof(T), __alignof(T))) T(t);
}

template<Class T> T* Allocator::allocateNewArr(Allocator& allocator, size_t length) {
	assert(length != 0);
	T* arr = (T*)allocator.allocate(sizeof(T)*length, __alignof(T));
	for (size_t i = 0; i < length; ++i) {
		*(arr + i) = new T();
	}
	return p;
}

template<Class T> void Allocator::deallocateItem(Allocator& allocator, T& object) {
	object.~T();
	allocator.deallocate(&object);
}

template<Class T> void Allocator::deallocateArr(Allocator& allocator, T* arr) {
	assert(arr != nullptr);
	//TODO deallocate all objects in the array
	allocator.deallocate(arr);
}
