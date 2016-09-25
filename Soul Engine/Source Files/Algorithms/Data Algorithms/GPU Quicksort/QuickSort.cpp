#include "QuickSort.h"	

#define BITONICSORT_LEN         1024            // Must be power of 2!
#define QSORT_MAXDEPTH          16              // Will force final bitonic stage at depth QSORT_MAXDEPTH+1


typedef struct __align__(128) qsortAtomicData_t
{
	volatile unsigned int lt_offset;    // Current output offset for <pivot
	volatile unsigned int gt_offset;    // Current output offset for >pivot
	volatile unsigned int sorted_count; // Total count sorted, for deciding when to launch next wave
	volatile unsigned int index;        // Ringbuf tracking index. Can be ignored if not using ringbuf.
} qsortAtomicData;

typedef struct qsortRingbuf_t
{
	volatile unsigned int head;         // Head pointer - we allocate from here
	volatile unsigned int tail;         // Tail pointer - indicates last still-in-use element
	volatile unsigned int count;        // Total count allocated
	volatile unsigned int max;          // Max index allocated
	unsigned int stacksize;             // Wrap-around size of buffer (must be power of 2)
	volatile void *stackbase;           // Pointer to the stack we're allocating from
} qsortRingbuf;

// Stack elem count must be power of 2!
#define QSORT_STACK_ELEMS   1*1024*1024 // One million stack elements is a HUGE number.

/*
void quicksort(T* data, int N)
{
  int i, j;
  T v, t;
 
  if( N <= 1 )
    return;
 
  // Partition elements
  v = data[0];
  i = 0;
  j = N;
  for(;;)
  {
    while(data[++i] < v && i < N) { }
    while(data[--j] > v) { }
    if( i >= j )
      break;
    t = data[i];
    data[i] = data[j];
    data[j] = t;
  }
  t = data[i-1];
  data[i-1] = data[0];
  data[0] = t;
  quicksort(data, i-1);
  quicksort(data+i, N-i);
}
*/

/*

Quicksort(A as array, low as int, high as int){
    if (low < high){
        pivot_location = Partition(A,low,high)
        Quicksort(A,low, pivot_location)
        Quicksort(A, pivot_location + 1, high)
    }
}
Partition(A as array, low as int, high as int){
     pivot = A[low]
     leftwall = low

     for i = low + 1 to high{
         if (A[i] < pivot) then{
             swap(A[i], A[leftwall])
             leftwall = leftwall + 1
         }
     }
     swap(pivot,A[leftwall])

    return (leftwall)}


*/