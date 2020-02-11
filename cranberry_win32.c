#include "cranberry_platform.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wingdi.h>

void cranpl_write_bmp(const char* fileName, uint8_t* pixels, uint32_t width, uint32_t height)
{
	const uint32_t stride = 4;

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

void* cranpl_create_window(const char* windowName, uint32_t width, uint32_t height)
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

bool cranpl_tick_window(void* windowHandle)
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

void cranpl_destroy_window(void* windowHandle)
{
	DestroyWindow(windowHandle);
}
