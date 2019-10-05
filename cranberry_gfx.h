#ifndef __CRANBERRY_GFX
#define __CRANBERRY_GFX

#include "cranberry_gfx_backend.h"

#define crang_max_material_input_count 4
#define crang_max_shader_layout_count 10

typedef struct _crang_gfx_t crang_gfx_t;

typedef struct
{
	float pos[3];
	float width;
	float height;
	float zoom;
} crang_camera_t;

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
		crang_index_type_e indexType;
	} indices;
} crang_mesh_desc_t;

typedef struct
{
	// Vertices aren't passed as strided input.
	// They're passed as one large chunk of memory to the shader.
	// This allows the shader to figure out how to stride it dynamically.
	crang_buffer_id_t vertInput;
	crang_shader_input_id_t vertShaderInput;

	crang_buffer_id_t indices;
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
		crang_shader_input_id_t inputs[crang_max_material_input_count];
		unsigned int count;
	} shaderInputs;

	struct
	{
		crang_shader_input_id_t inputs[crang_max_material_input_count];
		unsigned int count;
	} textureInputs;

	crang_pipeline_id_t pipeline;
} crang_material_t;

unsigned int crang_gfx_size(void);
crang_gfx_t* crang_create_gfx_win32(void* buffer, void* hinstance, void* hwnd);
void crang_destroy_gfx(crang_gfx_t* gfx);

crang_mesh_t crang_create_mesh(crang_gfx_t* gfx, crang_mesh_desc_t* meshDesc);
crang_shader_group_t crang_create_shader_group(crang_gfx_t* gfx, crang_shader_group_desc_t* shaderDesc);
crang_material_t crang_create_material(crang_gfx_t* gfx, crang_material_desc_t* matDesc);

#endif // __CRANBERRY_GFX

#ifdef CRANBERRY_GFX_IMPLEMENTATION

#include <stdint.h>
#include <string.h>

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

	gfx->layouts.defaultVertLayout = crang_request_shader_layout_id(gfx->backendDevice, crang_shader_flag_vertex);
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
							[0] = {.type = crang_shader_input_type_uniform_buffer, .binding = 0},
						},
						.count = 1,
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
	mesh.indices = crang_request_buffer_id(core->backendDevice);

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
					.type = crang_buffer_shader_input
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
					.type = crang_shader_input_type_uniform_buffer,
					.uniformBuffer = 
					{
						.bufferId = mesh.vertInput,
						.offset = 0,
						.size = sizeof(crang_vert_types_e) + meshDesc->verts.size
					}
				},
				&(crang_cmd_create_buffer_t)
				{
					.bufferId = mesh.indices,
					.size = meshDesc->indices.count,
					.type = crang_buffer_index
				},
				&(crang_cmd_copy_to_buffer_t)
				{
					.bufferId = mesh.indices,
					.data = meshDesc->indices.data,
					.size = meshDesc->indices.count,
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
		material.shaderInputs.buffers[i] = crang_request_buffer_id(core->backendDevice);
		material.shaderInputs.inputs[i] = crang_request_shader_input_id(core->backendDevice);

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
						.bufferId = material.shaderInputs.buffers[i],
						.size = matDesc->shaderInputs.inputBindings[i].size,
						.type = crang_buffer_shader_input
					},
					&(crang_cmd_copy_to_buffer_t)
					{
						.bufferId = material.shaderInputs.buffers[i],
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
							.bufferId = material.shaderInputs.buffers[i],
							.size = matDesc->shaderInputs.inputBindings[i].size,
							.offset = 0
						}
					}
				},
				.count = 4
			});
	}
	material.shaderInputs.count = matDesc->shaderInputs.count;

	for (uint32_t i = 0; i < matDesc->textureInputs.count; i++)
	{
		material.textureInputs.inputs[i] = crang_request_shader_input_id(core->backendDevice);

		promises[matDesc->shaderInputs.count + i] = crang_execute_commands_async(core->backendDevice,
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
						.shaderInputId = material.textureInputs.inputs[i]
					},
					&(crang_cmd_set_shader_input_data_t)
					{
						.shaderInputId = material.textureInputs.inputs[i],
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
	material.textureInputs.count = matDesc->textureInputs.count;

	for (uint32_t i = 0; i < matDesc->shaderInputs.count + matDesc->textureInputs.count; i++)
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


#endif // CRANBERRY_GFX_IMPLEMENTATION
