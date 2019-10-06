#ifndef __CRANBERRY_GFX
#define __CRANBERRY_GFX

#include "cranberry_gfx_backend.h"

#define crang_max_material_input_count 4
#define crang_max_shader_layout_count 10

typedef struct _crang_gfx_t crang_gfx_t;

typedef struct
{
	float pos[3];

	struct
	{
		float forward[3];
		float right[3];
		float up[3];
	} rotation;

	float width;
	float height;
	float zoom;
	float farPlane;
} crang_camera_t;

typedef struct
{
	float x;
	float y;
	float zA;
	float zB;
} crang_projection_t;

typedef struct
{
	float viewMat[16];
	crang_projection_t projection;
} crang_camera_transform_t;

typedef enum
{
	crang_vert_type_position_f32_3 = 0x01,
	crang_vert_type_max,
} crang_vert_types_e;

typedef struct
{
	struct
	{
		void* data;
		unsigned int size;

		crang_vert_types_e vertLayout;
	} verts;

	struct
	{
		void* data;
		unsigned int count;
		crang_index_type_e type;
	} indices;
} crang_mesh_desc_t;

typedef struct
{
	// Vertices aren't passed as strided input.
	// They're passed as one large chunk of memory to the shader.
	// This allows the shader to figure out how to stride it dynamically.
	crang_buffer_id_t vertInput;
	crang_shader_input_id_t vertShaderInput;

	struct
	{
		crang_buffer_id_t buffer;
		crang_index_type_e type;
		unsigned int count;
	} indices;
} crang_mesh_t;

typedef struct
{
	crang_shader_input_t* inputs;
	unsigned int count;
} crang_material_shader_layout_t;

typedef struct
{
	struct
	{
		void* source;
		unsigned int size;
	} vertShader;

	struct
	{
		void* source;
		unsigned int size;
	} fragShader;

	struct
	{
		crang_material_shader_layout_t* layouts;
		unsigned int count;
	} shaderLayouts;
} crang_shader_group_desc_t;

typedef struct
{
	crang_shader_id_t vertShader;
	crang_shader_id_t fragShader;
	
	struct
	{
		crang_shader_layout_id_t layouts[crang_max_shader_layout_count];
		unsigned int count;
	} shaderLayouts;
} crang_shader_group_t;

typedef struct
{
	void* data;
	unsigned int size;
	unsigned int binding;
	unsigned int shaderLayoutIndex; // Index into shader_group shader layouts
} crang_material_input_binding_t;

typedef struct
{
	crang_image_id_t texture;
	unsigned int binding;
	unsigned int shaderLayoutIndex; // Index into shader_group shader layouts
} crang_texture_input_binding_t;

typedef struct
{
	crang_shader_group_t* shaders;

	struct
	{
		crang_material_input_binding_t* inputBindings;
		unsigned int count;
	} shaderInputs;

	struct
	{
		crang_texture_input_binding_t* textureBindings;
		unsigned int count;
	} textureInputs;
} crang_material_desc_t;

typedef struct
{
	struct
	{
		crang_buffer_id_t buffers[crang_max_material_input_count];
		unsigned int count;
	} materialBuffers;

	struct
	{
		crang_shader_input_id_t inputs[crang_max_material_input_count];
		unsigned int shaderLayoutIndices[crang_max_material_input_count];
		unsigned int count;
	} shaderInputs;

	crang_pipeline_id_t pipeline;
} crang_material_t;

typedef struct
{
	crang_camera_transform_t* camera;
	struct
	{
		crang_mesh_t* meshes;
		unsigned int count;
	} meshGroups;
} crang_camera_group_t;

typedef struct
{
	crang_material_t* material;
	struct
	{
		crang_camera_group_t* cameras;
		unsigned int count;
	} cameraGroups;
} crang_material_group_t;

typedef struct
{
	struct
	{
		crang_material_group_t* materials;
		unsigned int count;
	} materialGroups;
} crang_draw_desc_t;

unsigned int crang_gfx_size(void);
crang_gfx_t* crang_create_gfx_win32(void* buffer, void* hinstance, void* hwnd);
void crang_destroy_gfx(crang_gfx_t* gfx);

crang_mesh_t crang_create_mesh(crang_gfx_t* gfx, crang_mesh_desc_t* meshDesc);
crang_shader_group_t crang_create_shader_group(crang_gfx_t* gfx, crang_shader_group_desc_t* shaderDesc);
crang_material_t crang_create_material(crang_gfx_t* gfx, crang_material_desc_t* matDesc);

crang_camera_transform_t crang_convert_camera_to_transform(crang_camera_t* camera);

crang_recording_buffer_id_t crang_record_draw(crang_gfx_t* gfx, crang_draw_desc_t* draw);
void crang_submit_draw(crang_gfx_t* gfx, crang_recording_buffer_id_t* recordings, unsigned int recordingCount);

#endif // __CRANBERRY_GFX

#ifdef CRANBERRY_GFX_IMPLEMENTATION

#define crang_max_recording_count 50
#define crang_max_draw_command_count 100

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#define crang_debug_enabled

#ifdef crang_debug_enabled
#define crang_assert(call) \
	do \
	{ \
		if (!(call)) \
		{ \
			__debugbreak(); \
		} \
	} while (0)
#else
#define crang_assert(call)
#endif // crang_debug_enabled

typedef struct
{
	crang_ctx_t* backendCtx;
	crang_surface_t* backendSurface;
	crang_graphics_device_t* backendDevice;
	crang_present_t* backendPresent;

	struct
	{
		crang_shader_layout_id_t defaultVertLayout;
	} layouts;

	struct
	{
		crang_recording_buffer_id_t recordings[crang_max_recording_count];
		bool allocated[crang_max_recording_count];
	} recordingPool;
} crang_gfx_core_t;

unsigned int crang_gfx_size(void)
{
	return sizeof(crang_gfx_core_t) + crang_ctx_size() + crang_win32_surface_size() + crang_graphics_device_size() + crang_present_size();
}

crang_gfx_t* crang_create_gfx_win32(void* buffer, void* hinstance, void* hwnd)
{
	uint8_t* gfxMem = (uint8_t*)buffer;
	memset(buffer, 0, crang_gfx_size());

	crang_gfx_core_t* gfx = (crang_gfx_core_t*)gfxMem;
	gfxMem += sizeof(crang_gfx_core_t);

	gfx->backendCtx = crang_create_ctx(gfxMem);
	gfxMem += crang_ctx_size();
	gfx->backendSurface = crang_win32_create_surface(gfxMem, gfx->backendCtx, hinstance, hwnd);
	gfxMem += crang_win32_surface_size();
	gfx->backendDevice = crang_create_graphics_device(gfxMem, gfx->backendCtx, gfx->backendSurface);
	gfxMem += crang_graphics_device_size();
	gfx->backendPresent = crang_create_present(gfxMem, gfx->backendDevice, gfx->backendSurface);
	gfxMem += crang_present_size();

	for (uint32_t i = 0; i < crang_max_recording_count; i++)
	{
		gfx->recordingPool.recordings[i] = crang_request_recording_buffer_id(gfx->backendDevice);
	}

	gfx->layouts.defaultVertLayout = crang_request_shader_layout_id(gfx->backendDevice, crang_shader_flag_vertex | crang_shader_flag_fragment);
	crang_execute_commands_immediate(gfx->backendDevice,
		&(crang_cmd_buffer_t)
		{
			.commandDescs = (crang_cmd_e[])
			{
				crang_cmd_create_shader_layout
			},
			.commandDatas = (void*[])
			{
				&(crang_cmd_create_shader_layout_t)
				{
					.shaderLayoutId = gfx->layouts.defaultVertLayout,
					.shaderInputs =
					{
						.inputs = (crang_shader_input_t[])
						{
							[0] = {.type = crang_shader_input_type_storage_buffer, .binding = 0},
						},
						.count = 1,
					},
					.immediateInputs = 
					{
						.inputs = (crang_immediate_input_t[])
						{
							[0] = { .offset = 0, .size = sizeof(crang_camera_transform_t) }
						},
						.count = 1
					}
				},
			},
			.count = 1
		});

	return (crang_gfx_t*)gfx;
}

void crang_destroy_gfx(crang_gfx_t* gfx)
{
	crang_gfx_core_t* gfxData = (crang_gfx_core_t*)gfx;
	crang_destroy_present(gfxData->backendDevice, gfxData->backendPresent);
	crang_destroy_graphics_device(gfxData->backendCtx, gfxData->backendDevice);
	crang_win32_destroy_surface(gfxData->backendCtx, gfxData->backendSurface);
	crang_destroy_ctx(gfxData->backendCtx);
}

crang_mesh_t crang_create_mesh(crang_gfx_t* gfx, crang_mesh_desc_t* meshDesc)
{
	crang_mesh_t mesh;

	crang_gfx_core_t* core = (crang_gfx_core_t*)gfx;

	mesh.vertInput = crang_request_buffer_id(core->backendDevice);
	mesh.vertShaderInput = crang_request_shader_input_id(core->backendDevice);
	mesh.indices.buffer = crang_request_buffer_id(core->backendDevice);
	mesh.indices.type = meshDesc->indices.type;
	mesh.indices.count = meshDesc->indices.count;

	uint32_t indexTypeSizeConversionTable[] =
	{
		[crang_index_type_u16] = 2,
		[crang_index_type_u32] = 4
	};

	crang_execute_commands_immediate(core->backendDevice,
		&(crang_cmd_buffer_t)
		{
			.commandDescs = (crang_cmd_e[])
			{
				crang_cmd_create_buffer,
				crang_cmd_copy_to_buffer,
				crang_cmd_copy_to_buffer,
				crang_cmd_create_shader_input,
				crang_cmd_set_shader_input_data,
				crang_cmd_create_buffer,
				crang_cmd_copy_to_buffer
			},
			.commandDatas = (void*[])
			{
				&(crang_cmd_create_buffer_t)
				{
					.bufferId = mesh.vertInput,
					.size = sizeof(crang_vert_types_e) + meshDesc->verts.size,
					.type = crang_buffer_shader_storage_input
				},
				&(crang_cmd_copy_to_buffer_t)
				{
					.bufferId = mesh.vertInput,
					.data = &meshDesc->verts.vertLayout,
					.size = sizeof(crang_vert_types_e),
					.offset = 0
				},
				&(crang_cmd_copy_to_buffer_t)
				{
					.bufferId = mesh.vertInput,
					.data = meshDesc->verts.data,
					.size = meshDesc->verts.size,
					.offset = sizeof(crang_vert_types_e)
				},
				&(crang_cmd_create_shader_input_t)
				{
					.shaderLayoutId = core->layouts.defaultVertLayout,
					.shaderInputId = mesh.vertShaderInput,
					.size = meshDesc->verts.size
				},
				&(crang_cmd_set_shader_input_data_t)
				{
					.shaderInputId = mesh.vertShaderInput,
					.binding = 0,
					.type = crang_shader_input_type_storage_buffer,
					.uniformBuffer = 
					{
						.bufferId = mesh.vertInput,
						.offset = 0,
						.size = sizeof(crang_vert_types_e) + meshDesc->verts.size
					}
				},
				&(crang_cmd_create_buffer_t)
				{
					.bufferId = mesh.indices.buffer,
					.size = meshDesc->indices.count * indexTypeSizeConversionTable[mesh.indices.type],
					.type = crang_buffer_index
				},
				&(crang_cmd_copy_to_buffer_t)
				{
					.bufferId = mesh.indices.buffer,
					.data = meshDesc->indices.data,
					.size = meshDesc->indices.count * indexTypeSizeConversionTable[mesh.indices.type],
					.offset = 0
				}
			},
			.count = 7
		});

	return mesh;
}

crang_shader_group_t crang_create_shader_group(crang_gfx_t* gfx, crang_shader_group_desc_t* shaderDesc)
{
	crang_gfx_core_t* core = (crang_gfx_core_t*)gfx;

	crang_shader_group_t shaders;

	shaders.vertShader = crang_request_shader_id(core->backendDevice, crang_shader_vertex);
	shaders.fragShader = crang_request_shader_id(core->backendDevice, crang_shader_fragment);
	crang_execute_commands_immediate(core->backendDevice, 
		&(crang_cmd_buffer_t)
		{
			.commandDescs = (crang_cmd_e[])
			{
				crang_cmd_create_shader,
				crang_cmd_create_shader
			},
			.commandDatas = (void*[])
			{
				&(crang_cmd_create_shader_t)
				{
					.shaderId = shaders.vertShader,
					.source = shaderDesc->vertShader.source,
					.sourceSize = shaderDesc->vertShader.size
				},
				&(crang_cmd_create_shader_t)
				{
					.shaderId = shaders.fragShader,
					.source = shaderDesc->fragShader.source,
					.sourceSize = shaderDesc->fragShader.size
				}
			},
			.count = 2
		});

	for (uint32_t i = 0; i < shaderDesc->shaderLayouts.count; i++)
	{
		shaders.shaderLayouts.layouts[i] = crang_request_shader_layout_id(core->backendDevice, crang_shader_flag_vertex | crang_shader_flag_fragment );
		crang_execute_commands_immediate(core->backendDevice, 
			&(crang_cmd_buffer_t)
			{
				.commandDescs = (crang_cmd_e[])
				{
					crang_cmd_create_shader_layout
				},
				.commandDatas = (void*[])
				{
					&(crang_cmd_create_shader_layout_t)
					{
						.shaderLayoutId = shaders.shaderLayouts.layouts[i],
						.shaderInputs = 
						{
							.inputs = shaderDesc->shaderLayouts.layouts[i].inputs,
							.count = shaderDesc->shaderLayouts.layouts[i].count,
						}
					},
				},
				.count = 1
			});
	}

	shaders.shaderLayouts.count = shaderDesc->shaderLayouts.count;

	return shaders;
}

crang_material_t crang_create_material(crang_gfx_t* gfx, crang_material_desc_t* matDesc)
{
	crang_gfx_core_t* core = (crang_gfx_core_t*)gfx;

	crang_material_t material;

	crang_promise_id_t promises[crang_max_material_input_count * 2];
	for (uint32_t i = 0; i < matDesc->shaderInputs.count; i++)
	{
		material.materialBuffers.buffers[i] = crang_request_buffer_id(core->backendDevice);
		material.shaderInputs.inputs[i] = crang_request_shader_input_id(core->backendDevice);
		material.shaderInputs.shaderLayoutIndices[i] = matDesc->shaderInputs.inputBindings[i].shaderLayoutIndex;

		promises[i] = crang_execute_commands_async(core->backendDevice,
			&(crang_cmd_buffer_t)
			{
				.commandDescs = (crang_cmd_e[])
				{
					crang_cmd_create_buffer,
					crang_cmd_copy_to_buffer,
					crang_cmd_create_shader_input,
					crang_cmd_set_shader_input_data
				},
				.commandDatas = (void*[])
				{
					&(crang_cmd_create_buffer_t)
					{
						.bufferId = material.materialBuffers.buffers[i],
						.size = matDesc->shaderInputs.inputBindings[i].size,
						.type = crang_buffer_shader_uniform_input
					},
					&(crang_cmd_copy_to_buffer_t)
					{
						.bufferId = material.materialBuffers.buffers[i],
						.data = matDesc->shaderInputs.inputBindings[i].data,
						.size = matDesc->shaderInputs.inputBindings[i].size,
						.offset = 0
					},
					&(crang_cmd_create_shader_input_t)
					{
						.shaderLayoutId = matDesc->shaders->shaderLayouts.layouts[matDesc->shaderInputs.inputBindings[i].shaderLayoutIndex],
						.shaderInputId = material.shaderInputs.inputs[i]
					},
					&(crang_cmd_set_shader_input_data_t)
					{
						.shaderInputId = material.shaderInputs.inputs[i],
						.binding = matDesc->shaderInputs.inputBindings[i].binding,
						.type = crang_shader_input_type_uniform_buffer,
						.uniformBuffer = 
						{
							.bufferId = material.materialBuffers.buffers[i],
							.size = matDesc->shaderInputs.inputBindings[i].size,
							.offset = 0
						}
					}
				},
				.count = 4
			});
	}
	material.materialBuffers.count = matDesc->shaderInputs.count;
	material.shaderInputs.count = matDesc->shaderInputs.count;

	for (uint32_t i = 0; i < matDesc->textureInputs.count; i++)
	{
		material.shaderInputs.inputs[i] = crang_request_shader_input_id(core->backendDevice);
		material.shaderInputs.shaderLayoutIndices[i] = matDesc->shaderInputs.inputBindings[i].shaderLayoutIndex;

		promises[material.materialBuffers.count + i] = crang_execute_commands_async(core->backendDevice,
			&(crang_cmd_buffer_t)
			{
				.commandDescs = (crang_cmd_e[])
				{
					crang_cmd_create_shader_input,
					crang_cmd_set_shader_input_data
				},
				.commandDatas = (void*[])
				{
					&(crang_cmd_create_shader_input_t)
					{
						.shaderLayoutId = matDesc->shaders->shaderLayouts.layouts[matDesc->shaderInputs.inputBindings[i].shaderLayoutIndex],
						.shaderInputId = material.shaderInputs.inputs[material.shaderInputs.count + i]
					},
					&(crang_cmd_set_shader_input_data_t)
					{
						.shaderInputId = material.shaderInputs.inputs[material.shaderInputs.count + i],
						.binding = matDesc->textureInputs.textureBindings[i].binding,
						.type = crang_shader_input_type_sampler,
						.sampler = 
						{
							.imageId = matDesc->textureInputs.textureBindings[i].texture
						}
					}
				},
				.count = 2
			});
	}
	material.shaderInputs.count += matDesc->textureInputs.count;

	for (uint32_t i = 0; i < matDesc->shaderInputs.count; i++)
	{
		crang_wait_promise(core->backendDevice, promises[i]);
	}

	crang_shader_layout_id_t shaderLayouts[crang_max_shader_layout_count];
	crang_assert(matDesc->shaders->shaderLayouts.count + 1 < crang_max_shader_layout_count);

	shaderLayouts[0] = core->layouts.defaultVertLayout;
	memcpy(shaderLayouts + 1, matDesc->shaders->shaderLayouts.layouts, matDesc->shaders->shaderLayouts.count * sizeof(crang_shader_layout_id_t));

	material.pipeline = crang_create_pipeline(core->backendDevice, &(crang_pipeline_desc_t)
	{
		.presentCtx = core->backendPresent,
		.shaders = 
		{
			[crang_shader_vertex] = matDesc->shaders->vertShader,
			[crang_shader_fragment] = matDesc->shaders->fragShader
		},
		.shaderLayouts = 
		{
			.layouts = shaderLayouts,
			.count = matDesc->shaders->shaderLayouts.count + 1
		},
		.vertexInputs = 
		{
			.count = 0
		},
		.vertexAttributes = 
		{
			.count = 0
		}
	});

	return material;
}

crang_camera_transform_t crang_convert_camera_to_transform(crang_camera_t* camera)
{
	float n = camera->zoom;
	float f = camera->farPlane;
	float w = camera->width;
	float h = camera->height;

	crang_camera_transform_t transform;
	memcpy(transform.viewMat,
		&(float[16])
		{
			camera->rotation.right[0], camera->rotation.right[1], camera->rotation.right[2], 0.0f,
			camera->rotation.up[0], camera->rotation.up[1], camera->rotation.up[2], 0.0f,
			camera->rotation.forward[0], camera->rotation.forward[1], camera->rotation.forward[2], 0.0f,
			camera->pos[0] - camera->rotation.forward[0] * n, camera->pos[1] - camera->rotation.forward[1] * n, camera->pos[2] - camera->rotation.forward[2] * n, 1.0f
		},
		sizeof(float) * 16);

	transform.projection.x = 2.0f * n / w;
	transform.projection.y = -2.0f * n / h;
	transform.projection.zA = f / (f - n);
	transform.projection.zB = -n * f / (f - n);

	return transform;
}

crang_recording_buffer_id_t crang_record_draw(crang_gfx_t* gfx, crang_draw_desc_t* draw)
{
	crang_gfx_core_t* core = (crang_gfx_core_t*)gfx;

	uint32_t commandIndex = 0;
	crang_cmd_e drawCommands[crang_max_draw_command_count];
	void* drawCommandDatas[crang_max_draw_command_count];

	crang_recording_buffer_id_t recording;
	for (uint32_t i = 0; i < crang_max_recording_count; i++)
	{
		if (!core->recordingPool.allocated[i])
		{
			recording = core->recordingPool.recordings[i];
			core->recordingPool.allocated[i] = true;
			break;
		}
	}

	// TODO: Assert that we found a recording

	crang_cmd_bind_pipeline_t cmdBindPipeline[crang_max_draw_command_count];

	uint32_t cmdBindInputIndex = 0;
	crang_cmd_bind_shader_input_t cmdBindShaderInput[crang_max_draw_command_count];

	uint32_t cmdCameraIndex = 0;
	crang_cmd_immediate_shader_input_t cmdImmediateShaderInput[crang_max_draw_command_count];

	uint32_t cmdBindIndicesIndex = 0;
	crang_cmd_bind_index_input_t cmdBindIndexInput[crang_max_draw_command_count];

	uint32_t cmdDrawIndicesIndex = 0;
	crang_cmd_draw_indexed_t cmdDrawIndexed[crang_max_draw_command_count];
	for (uint32_t matIndex = 0; matIndex < draw->materialGroups.count; matIndex++)
	{
		crang_material_t* material = draw->materialGroups.materials[matIndex].material;
		cmdBindPipeline[matIndex] = (crang_cmd_bind_pipeline_t)
		{
			.pipelineId = material->pipeline
		};

		drawCommands[commandIndex] = crang_cmd_bind_pipeline;
		drawCommandDatas[commandIndex] = &cmdBindPipeline[matIndex];
		commandIndex++;

		for (uint32_t inputIndex = 0; inputIndex < material->shaderInputs.count; inputIndex++)
		{
			cmdBindShaderInput[cmdBindInputIndex] = (crang_cmd_bind_shader_input_t)
			{
				.shaderLayoutIndex = material->shaderInputs.shaderLayoutIndices[inputIndex] + 1, // Add one, our first index is the mesh input
				.pipelineId = material->pipeline,
				.shaderInputId = material->shaderInputs.inputs[inputIndex]
			};

			drawCommands[commandIndex] = crang_cmd_bind_shader_input;
			drawCommandDatas[commandIndex] = &cmdBindShaderInput[cmdBindInputIndex];
			cmdBindInputIndex++;
			commandIndex++;
		}

		for (uint32_t cameraIndex = 0; cameraIndex < draw->materialGroups.materials[matIndex].cameraGroups.count; cameraIndex++)
		{
			crang_camera_transform_t* camera = draw->materialGroups.materials[matIndex].cameraGroups.cameras[cameraIndex].camera;

			cmdImmediateShaderInput[cmdCameraIndex] = (crang_cmd_immediate_shader_input_t)
			{
				.pipelineId = material->pipeline,
				.data = camera,
				.size = sizeof(crang_camera_transform_t),
				.supportedShaders = crang_shader_flag_vertex | crang_shader_flag_fragment
			};

			drawCommands[commandIndex] = crang_cmd_immediate_shader_input;
			drawCommandDatas[commandIndex] = &cmdImmediateShaderInput[cmdCameraIndex];
			cmdCameraIndex++;
			commandIndex++;

			for (uint32_t meshIndex = 0; meshIndex < draw->materialGroups.materials[matIndex].cameraGroups.cameras[cameraIndex].meshGroups.count; meshIndex++)
			{
				crang_mesh_t mesh = draw->materialGroups.materials[matIndex].cameraGroups.cameras[cameraIndex].meshGroups.meshes[meshIndex];

				cmdBindShaderInput[cmdBindInputIndex] = (crang_cmd_bind_shader_input_t)
				{
					.shaderLayoutIndex = 0,
					.pipelineId = material->pipeline,
					.shaderInputId = mesh.vertShaderInput
				};

				drawCommands[commandIndex] = crang_cmd_bind_shader_input;
				drawCommandDatas[commandIndex] = &cmdBindShaderInput[cmdBindInputIndex];
				cmdBindInputIndex++;
				commandIndex++;

				cmdBindIndexInput[cmdBindIndicesIndex] = (crang_cmd_bind_index_input_t)
				{
					.bufferId = mesh.indices.buffer,
					.offset = 0,
					.indexType = mesh.indices.type
				};

				drawCommands[commandIndex] = crang_cmd_bind_index_input;
				drawCommandDatas[commandIndex] = &cmdBindIndexInput[cmdBindIndicesIndex];
				cmdBindIndicesIndex++;
				commandIndex++;

				cmdDrawIndexed[cmdDrawIndicesIndex] = (crang_cmd_draw_indexed_t)
				{
					.indexCount = mesh.indices.count,
					.instanceCount = 1
				};

				drawCommands[commandIndex] = crang_cmd_draw_indexed;
				drawCommandDatas[commandIndex] = &cmdDrawIndexed[cmdDrawIndicesIndex];
				cmdDrawIndicesIndex++;
				commandIndex++;
			}
		}
	}

	crang_record_commands(core->backendDevice, core->backendPresent, recording,
		&(crang_cmd_buffer_t)
		{
			.commandDescs = drawCommands,
			.commandDatas = drawCommandDatas,
			.count = commandIndex
		});

	return recording;
}

void crang_submit_draw(crang_gfx_t* gfx, crang_recording_buffer_id_t* recordings, unsigned int recordingCount)
{
	crang_gfx_core_t* core = (crang_gfx_core_t*)gfx;

	crang_present(&(crang_present_desc_t)
		{
			.graphicsDevice = core->backendDevice,
			.presentCtx = core->backendPresent,
			.surface = core->backendSurface,
			.clearColor = { 0.1f, 0.1f, 0.1f },
			.recordedBuffers =
			{
				.buffers = recordings,
				.count = recordingCount
			}
		});
}


#endif // CRANBERRY_GFX_IMPLEMENTATION
