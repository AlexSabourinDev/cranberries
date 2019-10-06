#define _CRT_SECURE_NO_WARNINGS

#define CRANBERRY_GFX_IMPLEMENTATION
#include "../cranberry_gfx.h"

#include <stdbool.h>
#include <malloc.h>
#include <stdint.h>
#include <stdio.h>
#include <Windows.h>

void test_frontend(void* hinstance, void* hwnd)
{
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
					10.0f, 1.0f, 10.0f,
					0.0f, 1.0f, 1.0f,
					0.0f, -1.0f, 1.0f,
					10.0f, -1.0f, 10.0f
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
				.type = crang_index_type_u16,
				.count = 6
			}
		});

	void* vertSource;
	unsigned int vertSize;
	{
		FILE* file = fopen("test_cranberry_gfx_frontend_data/Shaders/SPIR-V/default.vspv", "rb");
		fseek(file, 0, SEEK_END);
		long fileSize = ftell(file);
		fseek(file, 0, SEEK_SET);

		vertSource = malloc(fileSize);
		vertSize = fileSize;
		fread(vertSource, fileSize, 1, file);
		fclose(file);
	}

	void* fragSource;
	unsigned int fragSize;
	{
		FILE* file = fopen("test_cranberry_gfx_frontend_data/Shaders/SPIR-V/default.fspv", "rb");
		fseek(file, 0, SEEK_END);
		long fileSize = ftell(file);
		fseek(file, 0, SEEK_SET);

		fragSource = malloc(fileSize);
		fragSize = fileSize;
		fread(fragSource, fileSize, 1, file);
		fclose(file);
	}

	crang_shader_group_t shaders = crang_create_shader_group(gfx,
		&(crang_shader_group_desc_t)
		{
			.vertShader = 
			{
				.source = vertSource,
				.size = vertSize
			},
			.fragShader = 
			{
				.source = fragSource,
				.size = fragSize
			},
			.shaderLayouts = 
			{
				.layouts = (crang_material_shader_layout_t[])
				{
					(crang_material_shader_layout_t)
					{
						.inputs = (crang_shader_input_t[])
						{
							{.type = crang_shader_input_type_uniform_buffer,.binding = 0 }
						},
						.count = 1
					},
					(crang_material_shader_layout_t)
					{
						.inputs = (crang_shader_input_t[])
						{
							{.type = crang_shader_input_type_sampler,.binding = 0 }
						},
						.count = 1
					}
				},
				.count = 2
			}
		});

	crang_material_t material = crang_create_material(gfx,
		&(crang_material_desc_t)
		{
			.shaders = &shaders,
			.shaderInputs = {0}
		});


	crang_camera_transform_t transform = crang_convert_camera_to_transform(
											&(crang_camera_t)
											{
												.pos = { 0.0f, 0.0f, -15.0f },
												.rotation = { .right = { 1.0f, 0.0f, 0.0f }, .up = { 0.0f, 1.0f, 0.0f }, .forward = { 0.0f, 0.0f, 1.0f } },
												.width = 16.0f,
												.height = 9.0f,
												.zoom = 10.0f,
												.farPlane = 1000.0f
											});
	crang_recording_buffer_id_t draw = crang_record_draw(gfx,
		&(crang_draw_desc_t)
		{
			.materialGroups = 
			{
				.materials = (crang_material_group_t[])

				{
					(crang_material_group_t)
					{
						.material = &material,
						.cameraGroups =
						{
							.cameras = (crang_camera_group_t[])
							{
								(crang_camera_group_t)
								{
									.camera = &transform,
									.meshGroups =
									{
										.meshes = (crang_mesh_t[])
										{
											mesh
										},
										.count = 1
									}
								}
							},
							.count = 1,
						}
					}
				},
				.count = 1
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

		crang_submit_draw(gfx, &draw, 1);
	}

	crang_destroy_gfx(gfx);
	free(gfxMem);
}
