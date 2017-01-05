#include "QuickSort.h"	

#define BITONICSORT_LEN         1024            // Must be power of 2!
#define QSORT_MAXDEPTH          16              // Will force final bitonic stage at depth QSORT_MAXDEPTH+1


//typedef struct __align__(128) qsortAtomicData_t
struct  qsortAtomicData_t
{
	volatile unsigned int lt_offset;    // Current output offset for <pivot
	volatile unsigned int gt_offset;    // Current output offset for >pivot
	volatile unsigned int sorted_count; // Total count sorted, for deciding when to launch next wave
	volatile unsigned int index;        // Ringbuf tracking index. Can be ignored if not using ringbuf.
} qsortAtomicData;

struct qsortRingbuf_t
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
	REFERENCES:
	http://www.cse.chalmers.se/~tsigas/papers/GPU-Quicksort-jea.pdf
	https://software.intel.com/en-us/articles/gpu-quicksort-in-opencl-20-using-nested-parallelism-and-work-group-scan-functions
	http://onlinelibrary.wiley.com/doi/10.1002/cpe.3611/abstract
*/
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