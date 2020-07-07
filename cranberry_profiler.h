#ifndef __CRANBERRY_PROFILER_H
#define __CRANBERRY_PROFILER_H

/*
cran_profiler Usage, License: MIT

About:
The cran_profiler is a single header utility that generates chrome:\\tracing json that can be then imported and viewed using chrome's
tracing utility. cran_profiler is completely thread safe and attempts to minimize contention between thread.

Usage:
Using cran_profiler is simple,

SETUP/TEARDOWN:
- #define CRANPR_IMPLEMENTATION before including the header file in one of your source files.
- #define CRANPR_ENABLE to enable profiling
- cranpr_init(); to init
- cranpr_terminate(); to close

NOTE: Chrome://tracing matches these calls by name and category assuring a unique name is important.

WARNING: Category and Name are not stored, their lifetime must exist either until program termination or until the next call to cranpr_flush_thread_buffer and cranpr_flush.

USAGE:
Once a significant amount of samples have been gathered, samples have to be flushed.
A simple way to do this is shown below

{
	// Buffers are only added to the list once they are full. This minimizes contention and allows simple modification of the buffers.
	// The buffers are also stored as thread local globals, and thus must be flushed from their respective threads.
	if(cranpr_captured_size() == 0)
	{
		// Adds the current buffer to the list of buffers even if it hasn't been filled up yet.
		cranpr_flush_thread_buffer();
	}

	char* print;
	size_t bufferSize;
	cranpr_flush_alloc(&print, &bufferSize);

	fprintf(fileHandle, "%s", cranpr_profile_preface);
	fprintf(fileHandle, "%s", print);
	fprintf(fileHandle, "%s", cranpr_profile_postface);

	free(print);
}

THREADING:

It is important to call cranpr_flush_thread_buffer() before shutting down a thread and 
before the last flush as the remaining buffers might have some samples left in them.

A multithreaded program could have the format:
{
	Init Profiler
	Add cranpr_profile_preface to the file

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
	Add cranpr_profile_postface to the file

	Terminate the profiler
}

*/

/* -API- */

#include <stdint.h>

#define cranpr_type_begin 'B'
#define cranpr_type_end 'E'
#define cranpr_type_instance 'I'

#ifdef CRANPR_ENABLED

#define cranpr_begin(cat, name) cranpr_write_sample(cranpr_create_sample(cat, name, cranpr_timestamp(), cranpr_type_begin));
#define cranpr_end(cat, name) cranpr_write_sample(cranpr_create_sample(cat, name, cranpr_timestamp(), cranpr_type_end));
#define cranpr_event(cat, name) cranpr_write_sample(cranpr_create_sample(cat, name, cranpr_timestamp(), cranpr_type_instance));

#else

#define cranpr_begin(cat, name)
#define cranpr_end(cat, name)
#define cranpr_event(cat, name)

#endif

typedef struct
{
	int64_t timeStamp;

	const char* category;
	const char* name;

	uint16_t processorID;
	uint16_t threadID;

	char eventType;

}  cranpr_sample_t;

static const char* cranpr_profile_preface = "{\"traceEvents\":[";
static const char* cranpr_profile_postface = "]}";

void cranpr_init(void);
void cranpr_terminate(void);

cranpr_sample_t cranpr_create_sample(const char* category, const char* name, int64_t timeStamp, char eventType);
void cranpr_write_sample(cranpr_sample_t sample);

uint32_t cranpr_captured_size(void);

size_t cranpr_string_size(void);

/* returns the written size */
size_t cranpr_flush(char* buffer, size_t* maxBufferSize);

/* returns the written size*/
size_t cranpr_flush_alloc(char** buffer, size_t* bufferSize);
void cranpr_free(char* buffer);
void cranpr_flush_thread_buffer(void);

int64_t cranpr_timestamp(void);

void cranpr_write_to_file(const char* filePath);

#endif /* __CRANBERRY_PROFILER_H */

/* -Implementation- */
#ifdef CRANPR_IMPLEMENTATION

/* -Platform Macros- */

#if defined(_WIN32) || defined(_WIN64) || defined(__WIN32__)
	#define CRANPR_WIN 1
#elif defined(macintosh) || defined(Macintosh) || defined(__APPLE__) && defined(__MACH__)
	#define CRANPR_MAC 1
#elif defined(__unix__) || defined(__unix)
	#define CRANPR_UNIX 1
#else
	#error "cranpr unsupported platform!"
#endif

#if defined(_MSC_VER)
	#define CRANPR_MSVC 1
#elif defined(__GNUC__)
	#define CRANPR_GCC 1
#else
	#error "cranpr unsupported compiler!"
#endif

#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if CRANPR_WIN
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

// grumble grumble near/far macros
#undef near
#undef far
#endif

#define cranpr_unused(a) (void)a

/* -Threads- */

#if CRANPR_MSVC
	#define CRANPR_THREAD_LOCAL __declspec( thread )
#else
	#error "cranpr thread local storage not implemented!"
#endif

#if CRANPR_WIN
	typedef CRITICAL_SECTION cranpr_lock_t;
#else
	#error "cranpr Mutex not implemented!"
#endif

void cranpr_init_lock(cranpr_lock_t* lock)
{
#if CRANPR_WIN
	InitializeCriticalSection(lock);
#else
	#error "cranpr_init_lock not implemented!"
#endif
}

void cranpr_terminate_lock(cranpr_lock_t* lock)
{
#if CRANPR_WIN
	DeleteCriticalSection(lock);
#else
	#error cranpr_terminate_lock not implemented!"
#endif
}

void cranpr_lock_section(cranpr_lock_t* lock)
{
#if CRANPR_WIN
	EnterCriticalSection(lock);
#else
	#error "cranpr_lock_section not implemented!"
#endif
}

void cranpr_unlock_section(cranpr_lock_t* lock)
{
#if CRANPR_WIN
	LeaveCriticalSection(lock);
#else
	#error "cranpr_unlock_section not implemented!"
#endif
}

uint16_t cranpr_get_thread_id( void )
{
#if CRANPR_WIN
	return (uint16_t)GetCurrentThreadId();
#else
	#error "cranpr_get_thread_id not implemented!"
#endif
}

uint16_t cranpr_get_process_id( void )
{
#if CRANPR_WIN
	return (uint16_t)GetProcessId(GetCurrentProcess());
#else
	#error "cranpr_get_process_id not implemented!"
#endif
}

/* -Timer- */

int64_t cranpr_timestamp( void )
{
#if CRANPR_WIN
	LARGE_INTEGER frequency;
	BOOL queryResult = FALSE;
	cranpr_unused(queryResult);

	queryResult = QueryPerformanceFrequency(&frequency);
	assert(queryResult == TRUE);

	LARGE_INTEGER time;
	queryResult = QueryPerformanceCounter(&time);
	assert(queryResult == TRUE);

	int64_t microSeconds = (int64_t)time.QuadPart * 1000000;
	return microSeconds / (int64_t)frequency.QuadPart;
#else
	#error "cranpr_timestamp not implemented!"
#endif
}

/* -Profiler- */

cranpr_sample_t cranpr_create_sample(const char* category, const char* name, int64_t timeStamp, char eventType)
{
	cranpr_sample_t sample;
	sample.timeStamp = timeStamp;
	sample.category = category;
	sample.name = name;
	sample.processorID = cranpr_get_process_id();
	sample.threadID = cranpr_get_thread_id();
	sample.eventType = eventType;
	return sample;
}

/* Bigger buffers mean less contention for the list, but also means longer flushes and more memory usage */
#define cranpr_buffer_size (1024)

typedef struct
{
	cranpr_sample_t samples[cranpr_buffer_size];
	uint16_t nextSampleWrite;

} cranpr_buffer_t;

typedef struct
{
	cranpr_buffer_t buffer;
	void* next;

} cranpr_buffer_node_t;


typedef struct
{
	cranpr_buffer_node_t* first;
	cranpr_buffer_node_t* last;
	uint32_t listSize;

	cranpr_lock_t lock;

} cranpr_buffer_list_t;

cranpr_buffer_list_t cranpr_buffer_list;
CRANPR_THREAD_LOCAL cranpr_buffer_t cranpr_buffer;

void cranpr_init( void )
{
	cranpr_init_lock(&cranpr_buffer_list.lock);
}

/* Terminate musst be the last thing called, assure that profiling events will no longer be called once this is called */
void cranpr_terminate( void )
{
	cranpr_buffer_node_t* iter;
	cranpr_lock_section(&cranpr_buffer_list.lock);

	iter = cranpr_buffer_list.first;
	cranpr_buffer_list.first = NULL;
	cranpr_buffer_list.last = NULL;
	cranpr_buffer_list.listSize = 0;

	cranpr_unlock_section(&cranpr_buffer_list.lock);

	cranpr_terminate_lock(&cranpr_buffer_list.lock);

	while (iter != NULL)
	{
		cranpr_buffer_node_t* next = (cranpr_buffer_node_t*)iter->next;
		free(iter);
		iter = next;
	}
}

uint32_t cranpr_captured_size(void)
{
	return cranpr_buffer_list.listSize;
}

/* Not thread safe */
// Call outside of lock
static cranpr_buffer_node_t* cranpr_create_node(cranpr_buffer_t* buffer)
{
	cranpr_buffer_node_t* node = (cranpr_buffer_node_t*)malloc(sizeof(cranpr_buffer_node_t));
	memcpy(&node->buffer, buffer, sizeof(cranpr_buffer_t));
	node->next = NULL;

	return node;
}

// Thread Safe
// Requires buffer to be allocated externally, try to do as little as possible outside of lock
static void cranpr_add_node_ts(cranpr_buffer_node_t* node)
{
	cranpr_lock_section(&cranpr_buffer_list.lock);
	if (cranpr_buffer_list.first == NULL)
	{
		assert(cranpr_buffer_list.last == NULL);
		cranpr_buffer_list.first = node;
		cranpr_buffer_list.last = node;
		cranpr_buffer_list.listSize = 1;
	}
	else
	{
		assert(cranpr_buffer_list.listSize != UINT32_MAX);
		assert(cranpr_buffer_list.last->next == NULL);

		cranpr_buffer_list.last->next = node;
		cranpr_buffer_list.last = node;
		cranpr_buffer_list.listSize++;
	}
	cranpr_unlock_section(&cranpr_buffer_list.lock);
}

/* Format: process Id, thread Id,  timestamp, event, category, name */
static const char* const cranpr_sample = "{\"pid\":%" PRIu16 ", \"tid\":%" PRIu16 ", \"ts\":%" PRId64 ", \"ph\":\"%c\", \"cat\": \"%s\", \"name\": \"%s\", \"args\":{\"t\":\"hi\"}},";

static size_t cranpr_sample_size(cranpr_sample_t* sample)
{
	size_t sampleSize = sizeof("{\"pid\":") - 1;
	sampleSize += sample->processorID == 0 ? 1 : (size_t)log10f((float)sample->processorID);
	sampleSize += sizeof(",\"tid\":") - 1;
	sampleSize += sample->threadID == 0 ? 1 : (size_t)log10f((float)sample->threadID);
	sampleSize += sizeof(",\"ts\":") - 1;
	sampleSize += sample->timeStamp == 0 ? 1 : (size_t)log10((double)sample->timeStamp);
	sampleSize += sizeof(",\"ph\":\"") - 1;
	sampleSize += 1; /* sample char */
	sampleSize += sizeof("\",\"cat\":\"") - 1;
	sampleSize += strlen(sample->category);
	sampleSize += sizeof("\", \"name\": \"") - 1;
	sampleSize += strlen(sample->name);
	sampleSize += sizeof("\", \"args\":{\"t\":\"hi\"}}") - 1;
	return sampleSize;
}

static void cranpr_reverse(char* start, char* end)
{
	end -= 1;
	for (; start < end; start++, end--)
	{
		char t = *start;
		*start = *end;
		*end = t;
	}
}

static void cranpr_write_u16(uint16_t val, char* writeBuffer, size_t* writePos)
{
	size_t start = *writePos;
	while (val >= 10)
	{
		// Avoid modulo for debug builds
		uint16_t t = val / 10;
		writeBuffer[(*writePos)++] = '0' + (char)(val - t * 10);
		val = t;
	}
	writeBuffer[(*writePos)++] = '0' + (char)val;

	cranpr_reverse(writeBuffer + start, writeBuffer + *writePos);
}

static void cranpr_write_i64(int64_t val, char* writeBuffer, size_t* writePos)
{
	size_t start = *writePos;
	while (val >= 10)
	{
		// Avoid modulo for debug builds
		int64_t t = val / 10;
		writeBuffer[(*writePos)++] = '0' + (char)(val - t * 10);
		val = t;
	}
	writeBuffer[(*writePos)++] = '0' + (char)val;

	cranpr_reverse(writeBuffer + start, writeBuffer + *writePos);
}

#define cranpr_memcpy_const_str(str, writeBuffer, writePos) \
	{ \
		memcpy(writeBuffer + *writePos, str, sizeof(str) - 1); \
		*writePos += sizeof(str) - 1; \
	}

static void cranpr_write_sample_to_string(cranpr_sample_t* sample, char* writeBuffer, size_t* writePos)
{
	cranpr_memcpy_const_str("{\"pid\":", writeBuffer, writePos);
	cranpr_write_u16(sample->processorID, writeBuffer, writePos);
	cranpr_memcpy_const_str(",\"tid\":", writeBuffer, writePos);
	cranpr_write_u16(sample->threadID, writeBuffer, writePos);
	cranpr_memcpy_const_str(",\"ts\":", writeBuffer, writePos);
	cranpr_write_i64(sample->timeStamp, writeBuffer, writePos);
	cranpr_memcpy_const_str(",\"ph\":\"", writeBuffer, writePos);
	writeBuffer[(*writePos)++] = sample->eventType;
	cranpr_memcpy_const_str("\",\"cat\":\"", writeBuffer, writePos);
	size_t strSize = strlen(sample->category);
	memcpy(writeBuffer + (*writePos), sample->category, strSize);
	(*writePos) += strSize;
	cranpr_memcpy_const_str("\",\"name\":\"", writeBuffer, writePos);
	strSize = strlen(sample->name);
	memcpy(writeBuffer + (*writePos), sample->name, strSize);
	(*writePos) += strSize;
	cranpr_memcpy_const_str("\",\"args\":{\"t\":\"hi\"}}", writeBuffer, writePos);
}

/* Calculates the size of the samples, allowing the memory to be allocated in one chunk */
/* Thread safe */
size_t cranpr_string_size(void)
{
	cranpr_buffer_node_t* start;
	/* We have to keep the lock while calculating the string size. We don't want the list to be stolen or modified while we're working. */
	cranpr_lock_section(&cranpr_buffer_list.lock);

	start = cranpr_buffer_list.first;

	size_t size = 0;
	while (start != NULL)
	{
		cranpr_buffer_node_t* next = (cranpr_buffer_node_t*)start->next;
		for (uint16_t i = 0; i < start->buffer.nextSampleWrite; i++)
		{
			cranpr_sample_t* sample = &start->buffer.samples[i];
			size += cranpr_sample_size(sample);
			size += (next != NULL || (i < start->buffer.nextSampleWrite - 1)) ? 1 : 0;
		}

		start = next;
	}

	cranpr_unlock_section(&cranpr_buffer_list.lock);
	
	return size + 1;
}

/* Returns a string to be printed, this takes time. */
/* Thread safe */
/* bufferSize must be at least >= cranpr_string_size(...) */
size_t cranpr_flush( char* buffer, size_t* bufferSize )
{
	assert(bufferSize != NULL);

	if (*bufferSize < 4)
	{
		assert(*bufferSize == 1);
		buffer[0] = '\0';
		return 1;
	}

	cranpr_buffer_node_t* start;
	cranpr_lock_section(&cranpr_buffer_list.lock);

	start = cranpr_buffer_list.first;
	cranpr_buffer_list.first = NULL;
	cranpr_buffer_list.last = NULL;
	cranpr_buffer_list.listSize = 0;

	cranpr_unlock_section(&cranpr_buffer_list.lock);

	if (start == NULL)
	{
		buffer[0] = '\0';
		return 1;
	}

	size_t size = 0;
	while (start != NULL)
	{
		cranpr_buffer_node_t* next = (cranpr_buffer_node_t*)start->next;
		for (uint16_t i = 0; i < start->buffer.nextSampleWrite; i++)
		{
			cranpr_sample_t* sample = &start->buffer.samples[i];
			cranpr_write_sample_to_string(sample, buffer, &size);

			if (next != NULL || (i < start->buffer.nextSampleWrite - 1))
			{
				buffer[size++] = ',';
			}
		}

		free(start);
		start = next;
	}

	if (size >= *bufferSize)
	{
		assert(false);
		return 0;
	}
	buffer[size] = '\0';
	return size + 1;
}

size_t cranpr_flush_alloc(char** buffer, size_t* bufferSize)
{
	*bufferSize = cranpr_string_size();
	*buffer = (char*)malloc(*bufferSize);

	return cranpr_flush(*buffer, bufferSize);
}

void cranpr_free(char* buffer)
{
	free(buffer);
}

/* Thread safe */
void cranpr_write_sample(cranpr_sample_t sample)
{
	cranpr_buffer.samples[cranpr_buffer.nextSampleWrite] = sample;

	cranpr_buffer.nextSampleWrite++;
	if (cranpr_buffer.nextSampleWrite == cranpr_buffer_size-1)
	{
		cranpr_add_node_ts(cranpr_create_node(&cranpr_buffer));
		cranpr_buffer.nextSampleWrite = 0;
	}
}

/* Thread safe */
void cranpr_flush_thread_buffer( void )
{
	cranpr_add_node_ts(cranpr_create_node(&cranpr_buffer));
	cranpr_buffer.nextSampleWrite = 0;
}

void cranpr_write_to_file(const char* filePath)
{
#ifdef WIN32
	FILE* fileHandle;
	errno_t error = fopen_s(&fileHandle, filePath, "w");
	if (error != 0)
	{
		return;
	}
#else
	FILE* fileHandle = fopen(filePath, "w");
	if (fileHandle == NULL)
	{
		return;
	}
#endif // WIN32

	char* print = 0;
	size_t bufferSize = 0;
	size_t writtenSize = cranpr_flush_alloc(&print, &bufferSize);

	if (writtenSize > 0)
	{
		fprintf(fileHandle, "%s", cranpr_profile_preface);
		fwrite(print, writtenSize - 1, 1, fileHandle);
		fprintf(fileHandle, "%s", cranpr_profile_postface);
	}

	cranpr_free(print);
	fclose(fileHandle);
}

#endif /* CRANPR_IMPLEMENTATION */
