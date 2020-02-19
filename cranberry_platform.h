#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef _MSC_BUILD
#define restrict __restrict
#endif

void cranpl_write_bmp(const char* restrict fileName, uint8_t* restrict pixels, uint32_t width, uint32_t height);
void* cranpl_create_window(const char* windowName, uint32_t width, uint32_t height);
bool cranpl_tick_window(void* windowHandle);
void cranpl_destroy_window(void* windowHandle);
uint64_t cranpl_timestamp_micro(void);
