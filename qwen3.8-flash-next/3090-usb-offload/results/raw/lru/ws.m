#import <Metal/Metal.h>
#include <stdio.h>
int main(){ id<MTLDevice> d=MTLCreateSystemDefaultDevice(); printf("recommendedMaxWorkingSetSize %.3f GiB, maxBufferLength %.3f GiB, current allocated %.3f GiB\n",(double)d.recommendedMaxWorkingSetSize/1073741824.0,(double)d.maxBufferLength/1073741824.0,(double)d.currentAllocatedSize/1073741824.0); return 0;}
