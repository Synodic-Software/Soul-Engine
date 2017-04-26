#include "stdafx.h"
#include "..\LinearAllocator.h"
#include"CppUnitTest.h"
#define MAX_MEMSIZE 512
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
/*This program performs unit tests to test accesses and modifications to memory managed by the allocator.
The tests are performed using Microsoft Visual Studio's Unit test framework.*/

TEST_CLASS(Linear_Allocator_Test) {
public:
	TEST_METHOD(Individual_Prim_Allocs) {
		/*Test allocation of individual primitive types*/
		int expected1 = 123;
		int expected2 = 456;
		size_t memSize = MAX_MEMSIZE;
		void* buffer = malloc(memSize);
		LinearAllocator linear(memSize, buffer);
		int* val1 = allocator::allocateNew<int>(linear); //test allocation without specific value
		*val1 = 123;
		int* val2 = allocator::allocateNew<int>(linear, expected2);
		Assert::AreEqual(expected1, *val1, L"Accessed incorrect value");
		Assert::AreEqual(expected2, *val2, L"Accessed incorrect value");
		linear.clear();
		free(buffer);
	}
	
	TEST_METHOD(Individual_Obj_Allocs) {
		/*Test allocation of individual objects*/
		std::string expectedStr1 = "Hello";
		std::string expectedStr2 = "HelloHi";
		size_t memSize = MAX_MEMSIZE;
		void* buffer = malloc(memSize);
		LinearAllocator linear(memSize, buffer);
		std::string* actStr1 = allocator::allocateNew<std::string>(linear);
		*actStr1 = expectedStr1;
		std::string* actStr2 = allocator::allocateNew<std::string>(linear, expectedStr2);
		Assert::AreEqual(expectedStr1, *actStr1, L"Accessed incorrect value.");
		Assert::AreEqual(expectedStr2, *actStr2, L"Accessed incorrect value.");
		linear.clear();
		free(buffer);
	}

	TEST_METHOD(Array_Prim_Allocs) {
		/*Test primitive array allocation*/
		size_t memSize = MAX_MEMSIZE;
		int numElements = 6;
		void* buffer = malloc(MAX_MEMSIZE);
		LinearAllocator linear(memSize, buffer);
		int* actArr = allocator::allocateNewArr<int>(linear, numElements);
		for (int i = 0; i < numElements; ++i) {
			*(actArr + i) = i;
		}
		/*Verify that the array is correct*/
		for (int i = 0; i < numElements; ++i) {
			Assert::AreEqual(i, *(actArr + i), L"Array not stored/accessed correctly.");
		}
		linear.clear();
		free(buffer);
	}

	TEST_METHOD(Array_Obj_Allocs) {
		/*Test object array allocation*/
		size_t memSize = MAX_MEMSIZE;
		int numElements = 6;
		std::string baseStr = "Hi";
		std::string arrStr = "";
		void* buffer = malloc(MAX_MEMSIZE);
		LinearAllocator linear(memSize, buffer);
		std::string* actArr = allocator::allocateNewArr<std::string>(linear, numElements);
		for (int i = 0; i < numElements; ++i) {
			arrStr += baseStr;
			*(actArr + i) = arrStr;
		}
		arrStr = "";
		/*Verify that the array is correct*/
		for (int i = 0; i < numElements; ++i) {
			arrStr += baseStr;
			Assert::AreEqual(arrStr, *(actArr + i), L"Array not stored/accessed correctly.");
		}
		linear.clear();
		free(buffer);
	}

	TEST_METHOD(Test_Capacity) {
		size_t memSize = 1;
		void* buffer = malloc(memSize);
		LinearAllocator linear(memSize, buffer);
		int* result = allocator::allocateNew<int>(linear);
		int* expected = nullptr;
		Assert::AreEqual(expected, result, L"Pointer should be null"); //LinearAllocator should not be able to allocate object larger than capacity
		linear.clear();
		free(buffer);
	}
};