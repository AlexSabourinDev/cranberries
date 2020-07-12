#include "cranberry_platform.h"

#include <assert.h>
#include <stdlib.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wingdi.h>
#include <shellapi.h>

void cranpl_write_bmp(char const* cran_restrict fileName, uint8_t* cran_restrict pixels, uint32_t width, uint32_t height)
{
	uint32_t const stride = 4;

	BITMAPFILEHEADER fileHeader = 
	{
		.bfType = 'MB', // BM inversed.
		.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + width * height * stride,
		.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) // This is bytes, no bits. Lies!
	};

	BITMAPINFOHEADER bitmapInfoHeader =
	{
		.biSize = sizeof(BITMAPINFOHEADER),
		.biWidth = width,
		.biHeight = height,
		.biPlanes = 1,
		.biBitCount = (WORD)stride * 8,
		.biCompression = BI_RGB,
		.biSizeImage = 0 // 0 for BI_RGB is fine
	};

	// Write!
	{
		HANDLE fileHandle = CreateFile(fileName, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

		DWORD bytesWritten;
		WriteFile(fileHandle, &fileHeader, sizeof(BITMAPFILEHEADER), &bytesWritten, NULL);
		WriteFile(fileHandle, &bitmapInfoHeader, sizeof(BITMAPINFOHEADER), &bytesWritten, NULL);
		WriteFile(fileHandle, pixels, width * height * stride, &bytesWritten, NULL);

		CloseHandle(fileHandle);
	}
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	switch(msg)
	{
		case WM_DESTROY:
			PostQuitMessage(0);
		break;
		default:
			return DefWindowProc(hwnd, msg, wParam, lParam);
	}
	return 0;
}

void* cran_restrict cranpl_create_window(char const* cran_restrict windowName, uint32_t width, uint32_t height)
{
	HINSTANCE hinstance = GetModuleHandle(NULL);
	RegisterClass(&(WNDCLASS)
	{
		.lpfnWndProc = WndProc,
		.hInstance = hinstance,
		.lpszClassName = windowName
	});

	HWND hwnd = CreateWindowEx(
		0,
		windowName,
		windowName,
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT, width, height,
		NULL,
		NULL,
		hinstance,
		NULL
		);
	ShowWindow(hwnd, SW_SHOWNORMAL);

	return hwnd;
}

bool cranpl_tick_window(void* cran_restrict windowHandle)
{
	(void)windowHandle;

	bool isDone = false;

	MSG msg;
	while(PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE))
	{
		if (msg.message == WM_QUIT)
		{
			isDone = true;
		}
		else
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}

	return isDone;
}

void cranpl_blit_bmp(void* window, uint8_t* cran_restrict pixels, uint32_t width, uint32_t height)
{
	HBITMAP map = CreateBitmap(width, height, 1, 8*4, pixels);
	HDC windowDC = GetDC(window);
	HDC src = CreateCompatibleDC(windowDC);
	SelectObject(src, map);

	RECT rect;
	GetWindowRect(window, &rect);
	StretchBlt(windowDC, 0, 0, rect.right - rect.left, rect.bottom - rect.top, src, 0, 0, width, height, SRCCOPY);
	DeleteDC(src);
}

void cranpl_destroy_window(void* cran_restrict windowHandle)
{
	DestroyWindow(windowHandle);
}

uint64_t cranpl_timestamp_micro( void )
{
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);
	LARGE_INTEGER time;
	QueryPerformanceCounter(&time);
	uint64_t microSeconds = (uint64_t)time.QuadPart * 1000000;
	return microSeconds / (uint64_t)frequency.QuadPart;
}

cranpl_file_map_t cranpl_map_file(char const* cran_restrict fileName)
{
	HANDLE fileHandle = CreateFile(fileName, GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (fileHandle == INVALID_HANDLE_VALUE)
	{
		return (cranpl_file_map_t){ 0 };
	}

	HANDLE fileMappingHandle = CreateFileMappingA(fileHandle, NULL, PAGE_READONLY, 0, 0, NULL);
	if (fileMappingHandle == NULL)
	{
		CloseHandle(fileHandle);
		return (cranpl_file_map_t){ 0 };
	}

	LPVOID map = MapViewOfFile(fileMappingHandle, FILE_MAP_READ, 0, 0, 0);
	if (map == NULL)
	{
		CloseHandle(fileMappingHandle);
		CloseHandle(fileHandle);
		return (cranpl_file_map_t){ 0 };
	}

	uint64_t fileSize = GetFileSize(fileHandle, NULL);
	return (cranpl_file_map_t) { ._handle1 = fileHandle, ._handle2 = fileMappingHandle, .fileData = map, .fileSize = fileSize };
}

void cranpl_unmap_file(cranpl_file_map_t fileMap)
{
	UnmapViewOfFile(fileMap.fileData);
	CloseHandle(fileMap._handle2);
	CloseHandle(fileMap._handle1);
}

typedef struct
{
	void(*function)(void*);
	void* data;
} thread_data_t;

static DWORD WINAPI platform_thread_callback(void* data)
{
	thread_data_t threadData = *(thread_data_t*)data;
	free(data);
	threadData.function(threadData.data);
	return 0;
}

uint32_t cranpl_get_core_count()
{
	SYSTEM_INFO systemInfo;
	GetSystemInfo(&systemInfo);

	return systemInfo.dwNumberOfProcessors;
}

void* cranpl_create_thread(void(*function)(void*), void* data)
{
	// TODO: custom allocator
	thread_data_t* threadData = malloc(sizeof(thread_data_t));
	*threadData = (thread_data_t)
	{
		.function = function,
		.data = data
	};
	HANDLE threadHandle = CreateThread(NULL, 0, &platform_thread_callback, threadData, 0, NULL);
	assert(threadHandle != NULL);

	return threadHandle;
}

void cranpl_wait_on_thread(void* threadHandle)
{
	WaitForSingleObjectEx(threadHandle, INFINITE, FALSE);
}

bool cranpl_wait_on_threads(void** threadHandles, uint32_t count, uint32_t waitTime)
{
	DWORD result = WaitForMultipleObjectsEx(count, threadHandles, TRUE, waitTime, FALSE);
	return result == WAIT_OBJECT_0;
}

long cranpl_atomic_increment(cranpl_atomic_int_t* atomic)
{
	assert((uintptr_t)&atomic->value % 4 == 0);
	return InterlockedIncrement(&atomic->value);
}

void cranpl_open_file_with_default_app(char const* cran_restrict fileName)
{
	ShellExecuteA(0, 0, fileName, 0, 0, SW_SHOW);
}

void cranpl_set_working_dir(char const* cran_restrict path)
{
	SetCurrentDirectory(path);
}
