
##############################################
#Memory

target_sources(${PROJECT_NAME}
    PRIVATE
        Allocator.cpp
        Allocator.h
        LinearAllocator.cpp
        LinearAllocator.h
        TaggedAllocator/TaggedAllocator.cpp
        TaggedAllocator/TaggedAllocator.h
        TaggedAllocator/TaggedHeap.cpp
        TaggedAllocator/TaggedHeap.h
)


