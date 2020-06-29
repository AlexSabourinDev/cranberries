#pragma warning(disable : 4204 4221)

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdbool.h>

#include <Windows.h>
#include "../cranberry_gfx.h"

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

int main(void)
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

	FILE* file = fopen("test_cranberry_gfx_data/Shaders/SPIR-V/default.vspv", "rb");
	fseek(file, 0, SEEK_END);
	long fileSize = ftell(file);
	fseek(file, 0, SEEK_SET);

	void* vertSource = malloc(fileSize);
	unsigned int vertSize = fileSize;
	fread(vertSource, fileSize, 1, file);
	fclose(file);

	file = fopen("test_cranberry_gfx_data/Shaders/SPIR-V/default.fspv", "rb");
	fseek(file, 0, SEEK_END);
	fileSize = ftell(file);
	fseek(file, 0, SEEK_SET);

	void* fragSource = malloc(fileSize);
	unsigned int fragSize = fileSize;
	fread(fragSource, fileSize, 1, file);
	fclose(file);

	crang_context_t* ctx = crang_init(&(crang_init_desc_t)
	{
		.win32 =
		{
			.hinstance = hinstance,
			.hwindow = hwnd
		},
		.materials =
		{
			.deferred = 
			{
				.gbufferVShader = vertSource,
				.gbufferVShaderSize = vertSize,
				.gbufferFShader = fragSource,
				.gbufferFShaderSize = fragSize
			}
		}
	});

	crang_image_id_t albedoImage = crang_create_image(ctx,
		&(crang_image_desc_t)
		{
			.format = crang_image_format_rgba8,
			.data = (uint8_t[])
			{
				0, 0, 255, 255,
				255, 0, 0, 255,
				0, 255, 0, 255,
				0, 255, 255, 255,
			},
			.width = 2,
			.height = 2,
		});
	crang_sampler_id_t albedoSampler = crang_create_sampler(ctx, &(crang_sampler_desc_t){0});

	crang_material_id_t someMaterial = crang_create_mat_deferred(ctx,
		&(crang_deferred_desc_t)
		{
			.albedoTint = { 0.0f, 1.0f, 1.0f, 1.0f },
			.albedoSampler = albedoSampler,
			.albedoImage = albedoImage,
		});

	crang_mesh_id_t someMesh = crang_create_mesh(ctx,
		&(crang_mesh_desc_t)
		{
			.vertices =
			{
				.data = (float[])
				{ 
					-1.0f, 1.0f, 0.0f,
					1.0f, 1.0f, 0.0f,
					1.0f, -1.0f, 0.0f,
					-1.0f, -1.0f, 0.0f,
				},
				.count = 4
			},
			.indices = 
			{
				.data = (uint32_t[])
				{
					0, 1, 2, 0, 2, 3,
				},
				.count = 2 * 3
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
			else if (msg.message == WM_PAINT)
			{
				crang_draw_view(ctx, 
					&(crang_view_t)
					{
						.viewProj = crang_mat4_identity,
						.batches[crang_material_deferred] = 
						{
							{
								.material = someMaterial,
								.instances = 
								{
									{
										.mesh = someMesh,
										.transforms = &crang_mat4_identity,
										.count = 1
									}
								}
							}
						}
					});
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
}
