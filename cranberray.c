#pragma warning(disable:4204) // Disable aggregate initializers warning. It's definitely valid in C.

#include <stdint.h>

#include <windows.h>
#include <wingdi.h>

static void cray_write_bmp(const char* fileName, uint8_t* pixels, uint32_t width, uint32_t height)
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

int main()
{
	uint8_t testBitmap[] =
	{
		0xFF,0x00,0x00,0x00,  0xFF,0x00,0x00,0x00,
		0x00,0xFF,0x00,0x00,  0x00,0xFF,0x00,0x00,
	};
	cray_write_bmp("render.bmp", testBitmap, 2, 2);
	return 0;
}
