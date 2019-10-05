#ifndef __CRANBERRY_GFX
#define __CRANBERRY_GFX

#include "cranberry_gfx_backend.h"

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
	// This allows the shader to figure out how to stride it.
	crang_buffer_id_t vertInput;
	crang_shader_input_id_t vertShaderInput;

	crang_buffer_id_t indices;
} crang_mesh_t;

unsigned int crang_gfx_size(void);
crang_gfx_t* crang_create_gfx_win32(void* buffer, void* hinstance, void* hwnd);
void crang_destroy_gfx(crang_gfx_t* gfx);

crang_mesh_t crang_create_mesh(crang_gfx_t* gfx, crang_mesh_desc_t* meshDesc);

#endif // __CRANBERRY_GFX 

#ifdef CRANBERRY_GFX_IMPLEMENTATION

#include <stdint.h>
#include <string.h>

typedef struct
{
	crang_ctx_t* backendCtx;
	crang_surface_t* backendSurface;
	crang_graphics_device_t* backendDevice;
	crang_present_t* backendPresent;

	struct
	{
		crang_promise_id_t layoutPromise;
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

	gfx->layouts.defaultVertLayout = crang_request_shader_layout_id(gfx->backendDevice, crang_shader_vertex);
	gfx->layouts.layoutPromise = crang_execute_commands_async(gfx->backendDevice,
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

	crang_wait_promise(core->backendDevice, core->layouts.layoutPromise);
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


#endif // CRANBERRY_GFX_IMPLEMENTATION
