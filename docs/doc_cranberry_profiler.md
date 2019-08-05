# cranberry_profiler
### About:
The cranberry_profiler is a single header utility that generates [chrome://tracing](chrome://tracing) json that can be then imported and viewed using chrome's tracing utility. You can also import non-empty traces at [speedscope](speedscope.app).
cranberry_profiler is thread safe.

### Usage:
Using `cranberry_profiler.h` is simple,
Include in one of your source files with define:
```C
	#define CRANPR_IMPLEMENTATION
	#define CRANPR_ENABLED // Must be included before all cranberry_profiler.h includes in order to allow the profiler to work.
	#include <cranberry_profiler.h>
```

Before profiling call:
```C
	cranpr_init();
```

And at the end of your program execution call
```C
	cranpr_terminate();
```
**Warning:** No other calls to the profiler should be made after terminate has been invoked!

To gather samples for the profiler, simply call

```C
	cranpr_begin("Category Of Sample", "Sample Name");

	// ...

	cranpr_end("Category Of Sample", "Sample Name");
```

[chrome://tracing](chrome://tracing) matches these calls by name and category. Determining a unique name for these is important.

**Warning:** `Category` and `Name` are not stored, their lifetime must exist either until program termination or until the next call to `cranpr_flush_thread_buffer` and `cranpr_flush`.

Once a significant amount of samples have been gathered, samples have to be flushed.
A simple way to do this is shown below
```C
	// Buffers are only added to the list once they are full.
	// This minimizes contention and allows simple modification of the buffers.
	// The buffers are also stored as thread local globals, and must be flushed from their threads.
	if(cranpr_captured_size() == 0)
	{
		// Adds the current buffer to the list of buffers even if it hasn't been filled up yet.
		cranpr_flush_thread_buffer();
	}

	char* print;
	size_t bufferSize;
	cranpr_flush_alloc(&print, &bufferSize);
	fprintf(fileHandle, "%s", print);
	free(print);
```
**Warning:** Flushing many samples can be slow, the size of the buffer can either be minimized or
flushing on a seperate thread can be used to minimize the latency.

Finally, when printing out the buffer, the set of samples must include the preface and postface.
```C
	fprintf(fileHandle, "%s", cranpr_preface);
	fprintf(fileHandle, "%s", flushedSamples);
	fprintf(fileHandle, ","); // We want a comma to merge flushes
	fprintf(fileHandle, "%s", moreFlushedSamples);
	fprintf(fileHandle, "%s", cranpr_postface);
```

### Threading:

It is important to call `cranpr_flush_thread_buffer()` before shutting down a thread and 
before the last flush as the remaining buffers might have some samples left in them.

A multithreaded program could have the format:
```
	#define CRANPR_IMPLEMENTATION
	#include <cranberry_profiler.h>
	
	Init Profiler
	Add canpr_preface to the file

	Startup job threads
		Create samples in these threads
		At thread termination, call cranpr_flush_thread_buffer()

	Startup flushing thread
		Call cranpr_captured_size()
		If there are buffers, flush them to a file
		At thread termination, call cranpr_flush_thread_buffer()

	Kill all the threads
	Call cranpr_flush_alloc() to flush the remaining buffers

	Print to a file
	Add cranpr_postface to the file

	Terminate the profiler
```

##### Tested Compilers
- MSVC 2017

##### Supported Platforms
- Windows

##### Planned Support
- Unix
- Mac OS
- GCC
- Clang

#### Sample Program
```C
#include <stdlib.h>
#include <stdio.h>

#define CRANPR_IMPLEMENTATION
#include <cranberry_profiler.h>

int main(int argc, char** argv)
{
	(void)argc;
	(void)argv;

	cranpr_init();

	FILE* file;
	fopen_s(&file, "Profiling.txt", "w");

	cranpr_begin("Full Profile", "Profiling...");
	for (int i = 0; i < 1024; i++)
	{
		cranpr_begin("Profiling!", "Profiling!");
		cranpr_end("Profiling!", "Profiling!");
		cranpr_event("Profiling!", "Event!");
	}
	cranpr_end("Full Profile", "Profiling...");
	cranpr_flush_thread_buffer();

	cranpr_begin("Profiling!", "Flushing!");
	char* buffer = cranpr_flush_alloc();
	cranpr_end("Profiling!", "Flushing!");

	cranpr_begin("Profiling!", "Writing To File!");
	fprintf(file, "%s", cranpr_preface);
	fprintf(file, "%s", buffer);
	free(buffer);
	cranpr_end("Profiling!", "Writing To File!");

	cranpr_flush_thread_buffer();
	buffer = cranpr_flush_alloc();

	fprintf(file, ", %s", buffer);
	fprintf(file, "%s", cranpr_postface);
	free(buffer);

	fclose(file);
	cranpr_terminate();
	return 0;
}
```

##### Thanks
Big thanks to
https://www.gamasutra.com/view/news/176420/Indepth_Using_Chrometracing_to_view_your_inline_profiling_data.php
https://aras-p.info/blog/2017/01/23/Chrome-Tracing-as-Profiler-Frontend/
and the team working on chrome://tracing or providing the tools and information needed to implement this library.

#### Change Log
- 2019-08-05: Moved from Mist_Profiler repo to cranberries in an attempt to coalesce single header libraries.
- 2019-02-02: Added MIST_PROFILE_ENABLED to allow the profiler to be enabled and disabled. This makes usage a bit more cumbersome, but simplifies turning off the compiler. This will break previous usage of the compiler.
- 2019-01-13: Changed Mist_Flush to Mist_FlushAlloc. Mist_Flush now requires an explicit buffer. This will break previous usage of the profiler.
- 2019-01-13: Changed MIST_BEGIN/END_PROFILE to MIST_PROFILE_BEGIN/END. Will break previous calls to old macros.
- 2018-02-14: Removed the need for a comma to append the buffers. This will break previous usage of the profiler but since it isn't a week old, I believe this is fine.
