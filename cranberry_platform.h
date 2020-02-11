#pragma once

#include <stdint.h>
#include <stdbool.h>

void cranpl_write_bmp(const char* fileName, uint8_t* pixels, uint32_t width, uint32_t height);
void* cranpl_create_window(const char* windowName, uint32_t width, uint32_t height);
bool cranpl_tick_window(void* windowHandle);
void cranpl_destroy_window(void* windowHandle);

