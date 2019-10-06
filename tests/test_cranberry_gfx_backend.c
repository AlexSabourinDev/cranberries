#define _CRT_SECURE_NO_WARNINGS

#define CRANBERRY_GFX_BACKEND_IMPLEMENTATION
#include "../cranberry_gfx_backend.h"

#include <malloc.h>
#include <stdio.h>

#include <Windows.h>

void test_backend(void* hinstance, void* hwnd)
{
	unsigned int ctxSize = crang_ctx_size();
	unsigned int surfaceSize = crang_win32_surface_size();
	unsigned int graphicsDeviceSize = crang_graphics_device_size();
	unsigned int presentCtxSize = crang_present_size();

	void* graphicsMemory = malloc(ctxSize + surfaceSize + graphicsDeviceSize + presentCtxSize);

	uint8_t* buffer = (uint8_t*)graphicsMemory;
	crang_ctx_t* ctx = crang_create_ctx(buffer);
	buffer += ctxSize;

	crang_surface_t* surface = crang_win32_create_surface(buffer, ctx, hinstance, hwnd);
	buffer += surfaceSize;

	crang_graphics_device_t* graphicsDevice = crang_create_graphics_device(buffer, ctx, surface);
	buffer += graphicsDeviceSize;

	crang_present_t* presentCtx = crang_create_present(buffer, graphicsDevice, surface);
	buffer += presentCtxSize;

	crang_shader_id_t vertShader = crang_request_shader_id(graphicsDevice, crang_shader_vertex);
	crang_shader_id_t fragShader = crang_request_shader_id(graphicsDevice, crang_shader_fragment);

	crang_shader_layout_id_t vertShaderLayout = crang_request_shader_layout_id(graphicsDevice, crang_shader_flag_vertex);
	crang_shader_layout_id_t fragShaderLayout = crang_request_shader_layout_id(graphicsDevice, crang_shader_flag_fragment);

	crang_promise_id_t vertPromise;
	crang_promise_id_t fragPromise;
	{
		FILE* file = fopen("test_cranberry_gfx_backend_data/Shaders/SPIR-V/default.vspv", "rb");
		fseek(file, 0, SEEK_END);
		long fileSize = ftell(file);
		fseek(file, 0, SEEK_SET);

		void* vertSource = malloc(fileSize);
		unsigned int vertSize = fileSize;
		fread(vertSource, fileSize, 1, file);
		fclose(file);

		vertPromise = crang_execute_commands_async(graphicsDevice,
			&(crang_cmd_buffer_t)
			{
				.commandDescs = (crang_cmd_e[])
				{
					[0] = crang_cmd_create_shader_layout,
					[1] = crang_cmd_create_shader,
					[2] = crang_cmd_callback
				},
				.commandDatas = (void*[])
				{
					[0] = &(crang_cmd_create_shader_layout_t)
					{
						.shaderLayoutId = vertShaderLayout,
						.shaderInputs =
						{
							.inputs = (crang_shader_input_t[])
							{
								[0] = {.type = crang_shader_input_type_uniform_buffer, .binding = 0},
							},
							.count = 1,
						}
					},
					[1] = &(crang_cmd_create_shader_t)
					{
						.shaderId = vertShader,
						.source = vertSource,
						.sourceSize = vertSize
					},
					[2] = &(crang_cmd_callback_t)
					{
						.callback = &free,
						.data = vertSource
					}
				},
				.count = 2
			});
	}

	{
		FILE* file = fopen("test_cranberry_gfx_backend_data/Shaders/SPIR-V/default.fspv", "rb");
		fseek(file, 0, SEEK_END);
		long fileSize = ftell(file);
		fseek(file, 0, SEEK_SET);

		void* fragSource = malloc(fileSize);
		unsigned int fragSize = fileSize;
		fread(fragSource, fileSize, 1, file);
		fclose(file);

		fragPromise = crang_execute_commands_async(graphicsDevice,
			&(crang_cmd_buffer_t)
			{
				.commandDescs = (crang_cmd_e[])
				{
					[0] = crang_cmd_create_shader_layout,
					[1] = crang_cmd_create_shader,
					[2] = crang_cmd_callback
				},
				.commandDatas = (void*[])
				{
					[0] = &(crang_cmd_create_shader_layout_t)
					{
						.shaderLayoutId = fragShaderLayout,
						.shaderInputs =
						{
							.inputs = (crang_shader_input_t[])
							{
								[0] = {.type = crang_shader_input_type_sampler, .binding = 1}
							},
							.count = 1,
						}
					},
					[1] = &(crang_cmd_create_shader_t)
					{
						.shaderId = fragShader,
						.source = fragSource,
						.sourceSize = fragSize
					},
					[2] = &(crang_cmd_callback_t)
					{
						.callback = &free,
						.data = fragSource
					}
				},
				.count = 2
			});
	}

	// Vertex Buffer
	crang_buffer_id_t vertexBuffer = crang_request_buffer_id(graphicsDevice);
	crang_buffer_id_t indexBuffer = crang_request_buffer_id(graphicsDevice);

	crang_promise_id_t meshBufferPromise;
	{
		meshBufferPromise = crang_execute_commands_async(graphicsDevice,
			&(crang_cmd_buffer_t)
			{
				.commandDescs = (crang_cmd_e[])
				{
					[0] = crang_cmd_create_buffer,
					[1] = crang_cmd_copy_to_buffer,
					[2] = crang_cmd_create_buffer,
					[3] = crang_cmd_copy_to_buffer
				},
				.commandDatas = (void*[])
				{
					[0] = &(crang_cmd_create_buffer_t)
					{
						.bufferId = vertexBuffer,
						.size = sizeof(float) * 3 * 8,
						.type = crang_buffer_vertex
					},
					[1] = &(crang_cmd_copy_to_buffer_t)
					{
						.bufferId = vertexBuffer,
						.data = (float[])
						{ 
							-1.0f, -1.0f, -1.0f,
							1.0f, -1.0f, -1.0f,
							1.0f, 1.0f, -1.0f,
							-1.0f, 1.0f, -1.0f,
							-1.0f, -1.0f, 1.0f,
							1.0f, -1.0f, 1.0f,
							1.0f, 1.0f, 1.0f,
							-1.0f, 1.0f, 1.0f
						},
						.size = sizeof(float) * 3 * 8,
						.offset = 0
					},
					[2] = &(crang_cmd_create_buffer_t)
					{
						.bufferId = indexBuffer,
						.size = sizeof(uint32_t) * 6 * 6,
						.type = crang_buffer_index
					},
					[3] = &(crang_cmd_copy_to_buffer_t)
					{
						.bufferId = indexBuffer,
						.data = (uint32_t[])
						{ 
							0, 1, 2, 0, 2, 3,
							1, 5, 6, 1, 6, 2,
							5, 4, 6, 5, 7, 6,
							0, 3, 4, 4, 3, 7,
							3, 2, 6, 3, 6, 7,
							0, 6, 1, 0, 4, 5
						},
						.size = sizeof(uint32_t) * 6 * 6,
						.offset = 0
					},
				},
				.count = 4
			});
	}

	// Images
	crang_image_id_t greyImage = crang_request_image_id(graphicsDevice);
	crang_promise_id_t imagePromise;
	{
		imagePromise = crang_execute_commands_async(graphicsDevice,
			&(crang_cmd_buffer_t)
			{
				.commandDescs = (crang_cmd_e[])
				{
					[0] = crang_cmd_create_image,
					[1] = crang_cmd_copy_to_image
				},
				.commandDatas = (void*[])
				{
					[0] = &(crang_cmd_create_image_t)
					{
						.imageId = greyImage,
						.format = crang_image_format_r8g8b8a8,
						.width = 2,
						.height = 2
					},
					[1] = &(crang_cmd_copy_to_image_t)
					{
						.imageId = greyImage,
						.format = crang_image_format_r8g8b8a8,
						.data = (uint8_t[])
						{
							0, 0, 255, 255,
							255, 0, 0, 255,
							0, 255, 0, 255,
							0, 255, 255, 255,
						},
						.offset = 0,
						.width = 2,
						.height = 2,
						.offsetX = 0,
						.offsetY = 0
					}
				},
				.count = 2
			});
	}

	// Uniforms
	typedef struct
	{
		float viewMatrix[16];
		float projectionMatrix[16];
	} camera_t;

	camera_t camera = { {0}, {0} };
	memcpy(&camera.viewMatrix, (float[16])
	{
		[0] = 1.0f,
		[3] = 9.0f,
		[5] = 1.0f,
		[10] = 1.0f,
		[11] = 15.0f,
		[15] = 1.0f
	}, sizeof(float) * 16);

	float l = -8.0f, r = 8.0f, t = 4.5f, b = -4.5f, n = 10.0f, f = 100.0f;
	memcpy(&camera.projectionMatrix, (float[16])
	{
		[0] = 2.0f * n / (r - l),
		[5] = -2.0f * n / (t - b),
		[10] = f / (f - n),
		[11] = -n * f / (f - n),
		[14] = 1.0f,
	}, sizeof(float) * 16);


	crang_wait_promise(graphicsDevice, vertPromise);
	crang_wait_promise(graphicsDevice, fragPromise);
	crang_wait_promise(graphicsDevice, meshBufferPromise);
	crang_wait_promise(graphicsDevice, imagePromise);

	crang_buffer_id_t vertInputBuffer = crang_request_buffer_id(graphicsDevice);
	crang_shader_input_id_t vertInputs = crang_request_shader_input_id(graphicsDevice);
	crang_shader_input_id_t fragSamplerInput = crang_request_shader_input_id(graphicsDevice);
	{
		crang_execute_commands_immediate(graphicsDevice,
			&(crang_cmd_buffer_t)
			{
				.commandDescs = (crang_cmd_e[])
				{
					[0] = crang_cmd_create_shader_input,
					[1] = crang_cmd_create_buffer,
					[2] = crang_cmd_copy_to_buffer,
					[3] = crang_cmd_set_shader_input_data,
					[4] = crang_cmd_create_shader_input,
					[5] = crang_cmd_set_shader_input_data
				},
				.commandDatas = (void*[])
				{
					[0] = &(crang_cmd_create_shader_input_t)
					{
						.shaderLayoutId = vertShaderLayout,
						.shaderInputId = vertInputs
					},
					[1] = &(crang_cmd_create_buffer_t)
					{
						.bufferId = vertInputBuffer,
						.size = sizeof(camera_t),
						.type = crang_buffer_shader_uniform_input
					},
					[2] = &(crang_cmd_copy_to_buffer_t)
					{
						.bufferId = vertInputBuffer,
						.data = &camera,
						.size = sizeof(camera_t),
						.offset = 0
					},
					[3] = &(crang_cmd_set_shader_input_data_t)
					{
						.shaderInputId = vertInputs,
						.binding = 0,
						.type = crang_shader_input_type_uniform_buffer,
						.uniformBuffer =
						{
							.bufferId = vertInputBuffer,
							.size = sizeof(camera_t),
							.offset = 0
						}
					},
					[4] = &(crang_cmd_create_shader_input_t)
					{
						.shaderLayoutId = fragShaderLayout,
						.shaderInputId = fragSamplerInput
					},
					[5] = &(crang_cmd_set_shader_input_data_t)
					{
						.shaderInputId = fragSamplerInput,
						.binding = 1,
						.type = crang_shader_input_type_sampler,
						.sampler = 
						{
							.imageId = greyImage
						}
					},
				},
				.count = 6
			});
	}

	crang_pipeline_id_t pipeline = crang_create_pipeline(graphicsDevice, &(crang_pipeline_desc_t)
	{
		.presentCtx = presentCtx,
		.shaders = 
		{
			[crang_shader_vertex] = vertShader,
			[crang_shader_fragment] = fragShader
		},

		.shaderLayouts = 
		{
			.layouts = (crang_shader_layout_id_t[])
			{
				vertShaderLayout, fragShaderLayout
			},
			.count = 2
		},

		.vertexInputs = 
		{
			.inputs = (crang_vertex_input_t[])
			{
				[0] = {.binding = 0,.stride = sizeof(float) * 3 }
			},
			.count = 1
		},

		.vertexAttributes = 
		{
			.attribs = (crang_vertex_attribute_t[])
			{
				[0] = {.binding = 0,.location = 0,.offset = 0,.format = crang_vertex_format_f32_3 }
			},
			.count = 1
		}
	});

	crang_recording_buffer_id_t rectangleDraw = crang_request_recording_buffer_id(graphicsDevice);
	crang_record_commands(graphicsDevice, presentCtx, rectangleDraw, 
		&(crang_cmd_buffer_t)
		{
			.commandDescs = (crang_cmd_e[])
			{
				[0] = crang_cmd_bind_pipeline,
				[1] = crang_cmd_bind_shader_input,
				[2] = crang_cmd_bind_shader_input,
				[3] = crang_cmd_bind_vertex_inputs,
				[4] = crang_cmd_bind_index_input,
				[5] = crang_cmd_draw_indexed,
			},
			.commandDatas = (void*[])
			{
				[0] = &(crang_cmd_bind_pipeline_t)
				{
					.pipelineId = pipeline
				},
				[1] = &(crang_cmd_bind_shader_input_t)
				{
					.shaderLayoutIndex = 0,
					.pipelineId = pipeline,
					.shaderInputId = vertInputs
				},
				[2] = &(crang_cmd_bind_shader_input_t)
				{
					.shaderLayoutIndex = 1,
					.pipelineId = pipeline,
					.shaderInputId = fragSamplerInput
				},
				[3] = &(crang_cmd_bind_vertex_inputs_t)
				{
					.bindings = (crang_vertex_input_binding_t[])
					{
						[0] = { .bufferId = vertexBuffer, .binding = 0, .offset = 0 }
					},
					.count = 1
				},
				[4] = &(crang_cmd_bind_index_input_t)
				{
					.bufferId = indexBuffer,
					.offset = 0,
					.indexType = crang_index_type_u32
				},
				[5] = &(crang_cmd_draw_indexed_t)
				{
					.indexCount = 36,
					.instanceCount = 1,
				}
			},
			.count = 6
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

		crang_present(&(crang_present_desc_t)
		{
			.graphicsDevice = graphicsDevice,
			.presentCtx = presentCtx,
			.surface = surface,
			.clearColor = { 0.7f, 0.2f, 0.1f },
			.recordedBuffers =
			{
				.buffers = (crang_recording_buffer_id_t[])
				{
					rectangleDraw
				},
				.count = 1
			}
		});
	}

	crang_destroy_present(graphicsDevice, presentCtx);
	crang_destroy_graphics_device(ctx, graphicsDevice);
	crang_win32_destroy_surface(ctx, surface);
	crang_destroy_ctx(ctx);
	free(graphicsMemory);
}
