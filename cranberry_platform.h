#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef _MSC_BUILD
#define cran_restrict __restrict
#else
#define cran_restrict restrict
#endif

void cranpl_write_bmp(char const* cran_restrict fileName, uint8_t* cran_restrict pixels, uint32_t width, uint32_t height);
void* cran_restrict cranpl_create_window(char const* windowName, uint32_t width, uint32_t height);
bool cranpl_tick_window(void* windowHandle);
void cranpl_destroy_window(void* windowHandle);

uint64_t cranpl_timestamp_micro(void);

typedef struct
{
	void* cran_restrict _handle1;
	void* cran_restrict _handle2;
	void* cran_restrict fileData;
	uint64_t fileSize;
} cranpl_file_map_t;
cranpl_file_map_t cranpl_map_file(char const* cran_restrict fileName);
void cranpl_unmap_file(cranpl_file_map_t fileMap);
