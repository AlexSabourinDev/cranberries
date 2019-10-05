
#define CRANBERRY_GFX_IMPLEMENTATION
#include "../cranberry_gfx.h"

#include <stdbool.h>
#include <malloc.h>
#include <stdint.h>
#include <Windows.h>

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch(msg)
    {
        case WM_CLOSE:
            DestroyWindow(hwnd);
        break;
        case WM_DESTROY:
            PostQuitMessage(0);
        break;
        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
    }
    return 0;
}

void test_frontend(void)
{
	HINSTANCE hinstance = GetModuleHandle(NULL);

	WNDCLASS wc = { 0 };
	wc.lpfnWndProc = WndProc;
	wc.hInstance = hinstance;
	wc.lpszClassName = "CranberryWindow";

	RegisterClass(&wc);

	HWND hwnd = CreateWindowEx(
		0,
		"CranberryWindow",
		"CRANBERRIES",
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
		NULL,
		NULL,
		hinstance,
		NULL
		);
	ShowWindow(hwnd, SW_SHOWNORMAL);

	unsigned int gfxSize = crang_gfx_size();
	void* gfxMem = malloc(gfxSize);

	crang_gfx_t* gfx = crang_create_gfx_win32(gfxMem, hinstance, hwnd);

	crang_mesh_t mesh = crang_create_mesh(gfx,
		&(crang_mesh_desc_t)
		{
			.verts = 
			{
				.data = (float[])
				{
					1.0f, 1.0f, 1.0f,
					0.0f, 1.0f, 1.0f,
					0.0f, 0.0f, 1.0f,
					1.0f, 0.0f, 1.0f
				},
				.size = sizeof(float) * 3 * 4,
				.vertLayout = crang_vert_type_position_f32_3
			},
			.indices =
			{
				.data = (uint16_t[])
				{
					0, 1, 2,
					0, 2, 3
				},
				.indexType = crang_index_type_u16,
				.count = 6
			}
		});

	while (true)
	{
		bool done = false;

		MSG msg;
		while(PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
			{
				done = true;
			}
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}

		if (done)
		{
			break;
		}
	}

	crang_destroy_gfx(gfx);
	free(gfxMem);
}
