#ifndef __CRANBERRY_GFX_BACKEND
#define __CRANBERRY_GFX_BACKEND

// Assumption log:
// - Someone is not going to bind a graphics device with an incompatible surface. That's alright for my use case where I will most likely only have
//   one device.

typedef struct _crang_ctx_t crang_ctx_t;
typedef struct _crang_graphics_device_t crang_graphics_device_t;
typedef struct _crang_surface_t crang_surface_t;
typedef struct _crang_present_t crang_present_t;

typedef struct { unsigned int id; } crang_shader_layout_id_t;
typedef struct { unsigned int id; } crang_shader_id_t;
typedef struct { unsigned int id; } crang_buffer_id_t;
typedef struct { unsigned int id; } crang_pipeline_id_t;
typedef struct { unsigned int id; } crang_recording_buffer_id_t;
typedef struct { unsigned int id; } crang_shader_input_id_t;
typedef struct { unsigned int id; } crang_image_id_t;
typedef struct { unsigned int id; } crang_promise_id_t;

typedef enum
{
	crang_vertex_format_f32_1,
	crang_vertex_format_f32_2,
	crang_vertex_format_f32_3,
	crang_vertex_format_max
} crang_vertex_format_e;

typedef struct
{
	unsigned int binding;
	unsigned int location;
	unsigned int offset;
	crang_vertex_format_e format;
} crang_vertex_attribute_t;

typedef struct
{
	unsigned int binding;
	unsigned int stride;
} crang_vertex_input_t;

typedef enum
{
	crang_shader_vertex = 0,
	crang_shader_fragment,
	crang_shader_max
} crang_shader_e;

typedef enum
{
	crang_shader_flag_vertex = 1 << crang_shader_vertex,
	crang_shader_flag_fragment = 1 << crang_shader_fragment
} crang_shader_flags_e;

typedef enum
{
	crang_buffer_vertex,
	crang_buffer_index,
	crang_buffer_shader_uniform_input,
	crang_buffer_shader_storage_input,
	crang_buffer_max
} crang_buffer_e;

typedef struct
{
	crang_present_t* presentCtx;

	crang_shader_id_t shaders[crang_shader_max];

	struct
	{
		crang_shader_layout_id_t* layouts;
		unsigned int count;
	} shaderLayouts;

	struct
	{
		crang_vertex_input_t* inputs;
		unsigned int count;
	} vertexInputs;

	struct
	{
		crang_vertex_attribute_t* attribs;
		unsigned int count;
	} vertexAttributes;

} crang_pipeline_desc_t;

typedef struct
{
	crang_graphics_device_t* graphicsDevice;
	crang_present_t* presentCtx;
	crang_surface_t* surface;
	float clearColor[3];

	struct
	{
		crang_recording_buffer_id_t* buffers;
		unsigned int count;
	} recordedBuffers;
} crang_present_desc_t;

typedef enum
{
	crang_cmd_create_shader,
	crang_cmd_create_shader_input,
	crang_cmd_create_shader_layout,
	crang_cmd_set_shader_input_data,
	crang_cmd_create_buffer,
	crang_cmd_copy_to_buffer,
	crang_cmd_create_image,
	crang_cmd_copy_to_image,
	crang_cmd_callback, // not sure about callbacks yet. I'll see if it bites me in the butt.
	crang_cmd_bind_pipeline,
	crang_cmd_bind_vertex_inputs,
	crang_cmd_bind_index_input,
	crang_cmd_bind_shader_input,
	crang_cmd_immediate_shader_input,
	crang_cmd_draw_indexed,
} crang_cmd_e;

typedef struct
{
	crang_cmd_e* commandDescs;
	void** commandDatas;
	unsigned int count;
} crang_cmd_buffer_t;

typedef enum
{
	crang_shader_input_type_uniform_buffer,
	crang_shader_input_type_storage_buffer,
	crang_shader_input_type_sampler,
} crang_shader_input_type_e;

typedef struct
{
	crang_shader_input_type_e type;
	unsigned int binding;
} crang_shader_input_t;

typedef struct
{
	unsigned int size;
	unsigned int offset;
} crang_immediate_input_t;

typedef struct
{
	crang_shader_id_t shaderId;
	void* source;
	unsigned int sourceSize;
} crang_cmd_create_shader_t;

typedef struct
{
	crang_shader_layout_id_t shaderLayoutId;

	struct
	{
		crang_shader_input_t* inputs;
		unsigned int count;
	} shaderInputs;

	crang_immediate_input_t immediateInput;
} crang_cmd_create_shader_layout_t;

typedef struct
{
	crang_shader_layout_id_t shaderLayoutId;
	crang_shader_input_id_t shaderInputId;
	unsigned int size;
} crang_cmd_create_shader_input_t;

typedef struct
{
	crang_shader_input_id_t shaderInputId;
	unsigned int binding;

	crang_shader_input_type_e type;
	union
	{
		struct
		{
			crang_buffer_id_t bufferId;
			unsigned int offset;
			unsigned int size;
		} uniformBuffer;

		struct
		{
			crang_image_id_t imageId;
		} sampler;
	};
} crang_cmd_set_shader_input_data_t;

typedef struct
{
	crang_buffer_id_t bufferId;
	unsigned int size;
	crang_buffer_e type;
} crang_cmd_create_buffer_t;

typedef struct
{
	crang_buffer_id_t bufferId;
	void* data;
	unsigned int size;
	unsigned int offset;
} crang_cmd_copy_to_buffer_t;

typedef enum
{
	crang_image_format_r8g8b8a8,
	crang_image_format_max
} crang_image_format_e;

typedef struct
{
	crang_image_id_t imageId;
	crang_image_format_e format;
	unsigned int width;
	unsigned int height;
} crang_cmd_create_image_t;

typedef struct
{
	crang_image_id_t imageId;
	crang_image_format_e format;
	void* data;
	unsigned int offset;
	unsigned int width;
	unsigned int height;
	unsigned int offsetX;
	unsigned int offsetY;
} crang_cmd_copy_to_image_t;

typedef struct
{
	crang_pipeline_id_t pipelineId;
} crang_cmd_bind_pipeline_t;

typedef struct
{
	crang_buffer_id_t bufferId;
	unsigned int binding;
	unsigned int offset;
} crang_vertex_input_binding_t;

typedef struct
{
	crang_vertex_input_binding_t* bindings;
	unsigned int count;
} crang_cmd_bind_vertex_inputs_t;

typedef enum
{
	crang_index_type_u16,
	crang_index_type_u32,
} crang_index_type_e;

typedef struct
{
	crang_index_type_e indexType;
	crang_buffer_id_t bufferId;
	unsigned int offset;
} crang_cmd_bind_index_input_t;

typedef struct
{
	unsigned int shaderLayoutIndex;
	crang_shader_input_id_t shaderInputId;
	crang_pipeline_id_t pipelineId;
} crang_cmd_bind_shader_input_t;

typedef struct
{
	crang_pipeline_id_t pipelineId;
	void* data;
	unsigned int size;
	unsigned int offset;
	crang_shader_flags_e supportedShaders;
} crang_cmd_immediate_shader_input_t;

typedef struct
{
	unsigned int indexCount;
	unsigned int indexOffset;
	unsigned int vertexOffset;
	unsigned int instanceCount;
} crang_cmd_draw_indexed_t;

typedef void(*crang_callback_t)(void* data);
typedef struct
{
	crang_callback_t callback;
	void* data;
} crang_cmd_callback_t;

unsigned int crang_ctx_size(void);
// buffer must be at least the size returned by crang_ctx_size
crang_ctx_t* crang_create_ctx(void* buffer);
void crang_destroy_ctx(crang_ctx_t* ctx);

unsigned int crang_win32_surface_size(void);
// buffer must be at least the size returned by crang_win32_surface_size
crang_surface_t* crang_win32_create_surface(void* buffer, crang_ctx_t* ctx, void* hinstance, void* window);
void crang_win32_destroy_surface(crang_ctx_t* ctx, crang_surface_t* surface);

unsigned int crang_graphics_device_size(void);
// buffer must be at least the size returned by crang_graphics_device_size
crang_graphics_device_t* crang_create_graphics_device(void* buffer, crang_ctx_t* ctx, crang_surface_t* surface);
void crang_destroy_graphics_device(crang_ctx_t* ctx, crang_graphics_device_t* device);

unsigned int crang_present_size(void);
// buffer must be at least the size returned by crang_present_ctx_size
crang_present_t* crang_create_present(void* buffer, crang_graphics_device_t* device, crang_surface_t* surface);
void crang_destroy_present(crang_graphics_device_t* device, crang_present_t* presentCtx);

crang_pipeline_id_t crang_create_pipeline(crang_graphics_device_t* device, crang_pipeline_desc_t* pipelineDesc);

crang_shader_layout_id_t crang_request_shader_layout_id(crang_graphics_device_t* device, crang_shader_flags_e supportedTypes);
crang_shader_id_t crang_request_shader_id(crang_graphics_device_t* device, crang_shader_e type);
crang_shader_input_id_t crang_request_shader_input_id(crang_graphics_device_t* device);
crang_buffer_id_t crang_request_buffer_id(crang_graphics_device_t* device);
crang_recording_buffer_id_t crang_request_recording_buffer_id(crang_graphics_device_t* device);
crang_image_id_t crang_request_image_id(crang_graphics_device_t* device);

// Execute a command stream immediately. Blocking call!
void crang_execute_commands_immediate(crang_graphics_device_t* device, crang_cmd_buffer_t* cmdBuffer);
crang_promise_id_t crang_execute_commands_async(crang_graphics_device_t* device, crang_cmd_buffer_t* cmdBuffer);
void crang_wait_promise(crang_graphics_device_t* device, crang_promise_id_t promise);

// Record a command stream, some commands might be executed in the process such as callbacks.
// Allows you to provide recorded commands to rendering.
void crang_record_commands(crang_graphics_device_t* device, crang_present_t* present, crang_recording_buffer_id_t recordingBuffer, crang_cmd_buffer_t* cmdBuffer);
void crang_present(crang_present_desc_t* renderDesc);

#endif // __CRANBERRY_BACKEND_GFX

#ifdef CRANBERRY_GFX_BACKEND_IMPLEMENTATION

#include <stddef.h>
#include <stdbool.h>

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#define VK_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#define cranvk_debug_enabled

#define cranvk_no_allocator NULL

#ifdef cranvk_debug_enabled
#define cranvk_check(call) \
	do \
	{ \
		if(VK_SUCCESS != (call)) \
		{ \
			__debugbreak(); \
		} \
	} while(0)

#define cranvk_assert(call) \
	do \
	{ \
		if (!(call)) \
		{ \
			__debugbreak(); \
		} \
	} while (0)

#define cranvk_error() \
	do \
	{ \
		__debugbreak(); \
	} while(0)
#else
#define cranvk_check(call) call
#define cranvk_assert(call)
#define cranvk_error()
#endif // cranvk_debug_enabled

#define cranvk_unused(a) (void)a

// Allocator

#define cranvk_max_allocator_pools 10
#define cranvk_max_memory_blocks 1000
#define cranvk_allocator_pool_size (1024 * 1024)

typedef struct
{
	VkDeviceSize size;
	VkDeviceSize offset;
	uint32_t id;
	uint32_t nextIndex;
	bool allocated;
} cranvk_memory_block_t;

typedef struct
{
	VkDeviceMemory memory;
	VkDeviceSize size;
	uint32_t headIndex;
	uint32_t nextId;
	uint32_t memoryType;
} cranvk_memory_pool_t;

typedef struct
{
	VkDeviceMemory memory;
	VkDeviceSize offset;
	uint32_t id;
	uint32_t poolIndex;
} cranvk_allocation_t;

typedef struct
{
	cranvk_memory_pool_t memoryPools[cranvk_max_allocator_pools];
	cranvk_memory_block_t blockPool[cranvk_max_memory_blocks];
	uint32_t freeBlocks[cranvk_max_memory_blocks];
	uint32_t freeBlockCount;
} cranvk_allocator_t;

static uint32_t cranvk_find_memory_index(VkPhysicalDevice physicalDevice, uint32_t typeBits, VkMemoryPropertyFlags requiredFlags, VkMemoryPropertyFlags preferedFlags)
{
	uint32_t preferedMemoryIndex = UINT32_MAX;
	VkPhysicalDeviceMemoryProperties physicalDeviceProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physicalDeviceProperties);

	VkMemoryType* types = physicalDeviceProperties.memoryTypes;

	for (uint32_t i = 0; i < physicalDeviceProperties.memoryTypeCount; ++i)
	{
		if ((typeBits & (1 << i)) && (types[i].propertyFlags & (requiredFlags | preferedFlags)) == (requiredFlags | preferedFlags))
		{
			return i;
		}
	}

	if (preferedMemoryIndex == UINT32_MAX)
	{
		for (uint32_t i = 0; i < physicalDeviceProperties.memoryTypeCount; ++i)
		{
			if ((typeBits & (1 << i)) && (types[i].propertyFlags & requiredFlags) == requiredFlags)
			{
				return i;
			}
		}
	}

	return UINT32_MAX;
}

static void cranvk_create_allocator(cranvk_allocator_t* allocator)
{
	memset(allocator->blockPool, 0xFF, sizeof(cranvk_memory_block_t) * cranvk_max_memory_blocks);
	memset(allocator->memoryPools, 0xFF, sizeof(cranvk_memory_pool_t) * cranvk_max_allocator_pools);

	allocator->freeBlockCount = cranvk_max_memory_blocks;
	for (uint32_t freeBlockIndex = 0; freeBlockIndex < allocator->freeBlockCount; freeBlockIndex++)
	{
		allocator->freeBlocks[freeBlockIndex] = freeBlockIndex;
	}
}

static void cranvk_destroy_allocator(VkDevice device, cranvk_allocator_t* allocator)
{
	for (unsigned int i = 0; i < cranvk_max_allocator_pools; i++)
	{
		uint32_t iter = allocator->memoryPools[i].headIndex;
		while (iter != UINT32_MAX)
		{
			allocator->freeBlocks[allocator->freeBlockCount] = iter;
			allocator->freeBlockCount++;

			cranvk_memory_block_t* currentBlock = &allocator->blockPool[iter];
			currentBlock->nextIndex = UINT32_MAX;
			iter = currentBlock->nextIndex;
		}

		if (allocator->memoryPools[i].headIndex != UINT32_MAX)
		{
			allocator->memoryPools[i].headIndex = UINT32_MAX;
			vkFreeMemory(device, allocator->memoryPools[i].memory, cranvk_no_allocator);
		}
	}
}

static cranvk_allocation_t cranvk_allocator_allocate(VkDevice device, cranvk_allocator_t* allocator, uint32_t memoryTypeIndex, VkDeviceSize size, VkDeviceSize alignment)
{
	cranvk_assert(memoryTypeIndex != UINT32_MAX);

	uint32_t foundPoolIndex = UINT32_MAX;
	for (uint32_t i = 0; i < cranvk_max_allocator_pools; i++)
	{
		if (allocator->memoryPools[i].memoryType == memoryTypeIndex)
		{
			foundPoolIndex = i;
			break;
		}
	}

	if (foundPoolIndex == UINT32_MAX)
	{
		for (unsigned int i = 0; i < cranvk_max_allocator_pools; i++)
		{
			cranvk_memory_pool_t* memoryPool = &allocator->memoryPools[i];
			if (memoryPool->headIndex == UINT32_MAX)
			{
				memoryPool->size = cranvk_allocator_pool_size;
				VkMemoryAllocateInfo memoryAllocInfo =
				{
					.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
					.allocationSize = memoryPool->size,
					.memoryTypeIndex = memoryTypeIndex
				};

				cranvk_check(vkAllocateMemory(device, &memoryAllocInfo, cranvk_no_allocator, &memoryPool->memory));

				cranvk_assert(allocator->freeBlockCount > 0);
				uint32_t newBlockIndex = allocator->freeBlocks[allocator->freeBlockCount - 1];
				allocator->freeBlockCount--;

				cranvk_memory_block_t* block = &allocator->blockPool[newBlockIndex];
				block->size = memoryPool->size;
				block->offset = 0;
				block->allocated = false;

				memoryPool->headIndex = newBlockIndex;
				memoryPool->nextId = 1;
				memoryPool->memoryType = memoryTypeIndex;

				foundPoolIndex = i;
				break;
			}
		}
	}

	cranvk_assert(foundPoolIndex != UINT32_MAX);

	// Fun little trick to round to next nearest power of 2 from https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
	// Reduce number by 1, will handle powers of 2
	size--;
	// Set all the lower bits to get a full set bit pattern giving us Pow2 - 1
	size |= size >> 1;
	size |= size >> 2;
	size |= size >> 4;
	size |= size >> 8;
	size |= size >> 16;
	// Add 1 to push to pow of 2
	size++;
	VkDeviceSize allocationSize = size + (alignment - size % alignment);

	cranvk_memory_pool_t* memoryPool = &allocator->memoryPools[foundPoolIndex];
	// Look for a free block our size
	for (uint32_t iter = memoryPool->headIndex; iter != UINT32_MAX; iter = allocator->blockPool[iter].nextIndex)
	{
		cranvk_memory_block_t* memoryBlock = &allocator->blockPool[iter];
		if (!memoryBlock->allocated && memoryBlock->size == allocationSize)
		{
			memoryBlock->allocated = true;
			return (cranvk_allocation_t)
			{
				.memory = memoryPool->memory,
				.offset = memoryBlock->offset,
				.id = memoryBlock->id,
				.poolIndex = foundPoolIndex
			};
		}
	}

	// Couldn't find a block the right size, create one from our closest block
	cranvk_memory_block_t* smallestBlock = NULL;
	for (uint32_t iter = memoryPool->headIndex; iter != UINT32_MAX; iter = allocator->blockPool[iter].nextIndex)
	{
		cranvk_memory_block_t* block = &allocator->blockPool[iter];

		if (smallestBlock == NULL || (block->size > allocationSize && block->size < smallestBlock->size && !block->allocated))
		{
			smallestBlock = block;
		}
	}

	cranvk_memory_block_t* iter = smallestBlock;
	if (iter == NULL)
	{
		cranvk_error();
		return (cranvk_allocation_t) { 0 };
	}

	while (iter->size != allocationSize)
	{
		VkDeviceSize newBlockSize = iter->size / 2;

		iter->allocated = true;
		cranvk_assert(allocator->freeBlockCount >= 2);

		uint32_t leftIndex = allocator->freeBlocks[allocator->freeBlockCount - 1];
		allocator->freeBlockCount--;

		cranvk_memory_block_t* left = &allocator->blockPool[leftIndex];
		left->offset = iter->offset;
		left->size = newBlockSize;
		left->id = memoryPool->nextId;
		left->allocated = false;
		++memoryPool->nextId;

		uint32_t rightIndex = allocator->freeBlocks[allocator->freeBlockCount - 1];
		allocator->freeBlockCount--;

		cranvk_memory_block_t* right = &allocator->blockPool[rightIndex];
		right->offset = iter->offset + newBlockSize;
		right->size = newBlockSize;
		right->id = memoryPool->nextId;
		right->allocated = false;
		++memoryPool->nextId;


		left->nextIndex = rightIndex;
		right->nextIndex = iter->nextIndex;
		iter->nextIndex = leftIndex;

		iter = left;
	}

	iter->allocated = true;
	return (cranvk_allocation_t)
	{
		.memory = memoryPool->memory,
		.offset = iter->offset,
		.id = iter->id,
		.poolIndex = foundPoolIndex
	};
}

static void cranvk_allocator_free(cranvk_allocator_t* allocator, cranvk_allocation_t allocation)
{
	cranvk_memory_pool_t* memoryPool = &allocator->memoryPools[allocation.poolIndex];

	uint32_t prevIters[2] = { UINT32_MAX, UINT32_MAX };

	for (uint32_t iter = memoryPool->headIndex; iter != UINT32_MAX; iter = allocator->blockPool[iter].nextIndex)
	{
		cranvk_memory_block_t* currentBlock = &allocator->blockPool[iter];
		if (currentBlock->id == allocation.id)
		{
			currentBlock->allocated = false;

			// We can't have a sibling to merge if there was never a previous iterator. This is because
			// the first previous iterator would be the root block that has no siblings
			if (prevIters[0] != UINT32_MAX)
			{
				cranvk_memory_block_t* previousBlock = &allocator->blockPool[prevIters[0]];
				// Previous iterator is my size, it's my sibling. If it's not allocated, merge it
				if (previousBlock->size == currentBlock->size && !previousBlock->allocated)
				{
					cranvk_memory_block_t* parentBlock = &allocator->blockPool[prevIters[1]];
					parentBlock->allocated = false;
					parentBlock->nextIndex = currentBlock->nextIndex;

					allocator->freeBlocks[allocator->freeBlockCount] = iter;
					allocator->freeBlocks[allocator->freeBlockCount + 1] = prevIters[0];
					allocator->freeBlockCount += 2;
				}
				// Since we just checked to see if the previous iterator was our sibling and it wasnt
				// we know that if we have a next iterator, it's our sibling
				else if (currentBlock->nextIndex != UINT32_MAX)
				{
					cranvk_memory_block_t* nextBlock = &allocator->blockPool[currentBlock->nextIndex];
					if (!nextBlock->allocated)
					{
						cranvk_memory_block_t* parentBlock = &allocator->blockPool[prevIters[0]];

						parentBlock->allocated = false;
						parentBlock->nextIndex = nextBlock->nextIndex;

						allocator->freeBlocks[allocator->freeBlockCount] = currentBlock->nextIndex;
						allocator->freeBlocks[allocator->freeBlockCount + 1] = iter;
						allocator->freeBlockCount += 2;
					}
				}
			}
			break;
		}

		prevIters[1] = prevIters[0];
		prevIters[0] = iter;
	}
}

// Main Rendering

// TODO: Reference to Windows, if we want multiplatform we'll have to change this.
#ifdef cranvk_debug_enabled
#define cranvk_instance_extension_count 3
const char* cranvk_instance_extensions[cranvk_instance_extension_count] = { VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME, VK_EXT_DEBUG_REPORT_EXTENSION_NAME };
#else
#define cranvk_instance_extension_count 2
const char* cranvk_instance_extensions[cranvk_instance_extension_count] = { VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME };
#endif // cranvk_debug_enabled

#define cranvk_device_extension_count 1
const char* cranvk_device_extensions[cranvk_device_extension_count] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

#ifdef cranvk_debug_enabled
#define cranvk_validation_count 1
const char* cranvk_validation_layers[cranvk_validation_count] = { "VK_LAYER_LUNARG_standard_validation" };
#else
#define cranvk_validation_count 0
const char* cranvk_validation_layers[cranvk_validation_count] = {};
#endif

#define cranvk_render_buffer_count 2
#define cranvk_max_physical_device_count 8
#define cranvk_max_physical_device_property_count 50
#define cranvk_max_physical_image_count 10
#define cranvk_max_uniform_buffer_count 1000
#define cranvk_max_storage_buffer_count 1000
#define cranvk_max_image_sampler_count 1000
#define cranvk_max_descriptor_set_count 1000
#define cranvk_max_shader_count 100
#define cranvk_max_shader_layout_count 50
#define cranvk_max_buffer_count 100
#define cranvk_max_framebuffer_count 100
#define cranvk_max_single_use_resource_count 10
#define cranvk_max_pipeline_count 10
#define cranvk_max_command_buffer_count 1000
#define cranvk_max_shader_inputs 32
#define cranvk_max_vertex_inputs 32
#define cranvk_max_vertex_attributes 32
#define cranvk_max_image_count 100
#define cranvk_max_promise_count 100
#define cranvk_max_pipeline_shader_layout_count 10
#define cranvk_max_immediate_input_count 10

typedef struct
{
	VkInstance instance;
} cranvk_ctx_t;

typedef struct
{
	VkSurfaceKHR surface;
} cranvk_surface_t;

typedef struct
{
	VkBuffer buffers[cranvk_max_single_use_resource_count];
	uint32_t bufferCount;
	cranvk_allocation_t allocations[cranvk_max_single_use_resource_count];
	uint32_t allocationCount;
} cranvk_transient_resources_t;

typedef struct
{
	VkCommandBuffer commandBuffer;
	
	// Temp resources are deallocated when an execution context is closed
	cranvk_transient_resources_t singleUseResources;
} cranvk_execution_ctx_t;

typedef struct
{
	struct
	{
		VkPhysicalDevice physicalDevice;
		VkDevice logicalDevice;
	} devices;

	struct
	{
		uint32_t graphicsQueueIndex;
		uint32_t presentQueueIndex;
		VkQueue presentQueue;
		VkQueue graphicsQueue;
	} queues;

	struct
	{
		crang_shader_flags_e supportedTypes[cranvk_max_shader_layout_count];
		VkDescriptorSetLayout descriptorSetLayouts[cranvk_max_shader_layout_count];

		crang_immediate_input_t immediateInputs[cranvk_max_shader_layout_count]; // Only a single immediate input is supported per shader layout.

		uint32_t layoutCount;
	} shaderLayouts;

	struct
	{
		crang_shader_e types[cranvk_max_shader_count];
		VkShaderModule shaders[cranvk_max_shader_count];

		struct
		{
			VkDescriptorSet sets[cranvk_max_descriptor_set_count];
			uint32_t count;
		} descriptorSets;

		uint32_t shaderCount;
	} shaders;

	struct
	{
		VkBuffer buffers[cranvk_max_buffer_count];
		cranvk_allocation_t allocations[cranvk_max_buffer_count];
		uint32_t bufferCount;
	} buffers;

	struct
	{
		VkPipeline pipelines[cranvk_max_pipeline_count];
		VkPipelineLayout layouts[cranvk_max_pipeline_count];
		uint32_t pipelineCount;
	} pipelines;

	struct
	{
		// Recording buffers keep track of their single use resources until they're reset.
		cranvk_transient_resources_t singleUseResources[cranvk_max_command_buffer_count];
		VkCommandBuffer recordingBuffers[cranvk_max_command_buffer_count];
		uint32_t bufferCount;
	} commandBuffers;

	struct
	{
		VkFramebuffer framebuffers[cranvk_max_framebuffer_count];
		uint32_t count;
	} framebuffers;

	struct
	{
		VkImage images[cranvk_max_image_count];
		VkImageView imageViews[cranvk_max_image_count];
		VkSampler samplers[cranvk_max_image_count];
		cranvk_allocation_t allocations[cranvk_max_image_count];
		uint32_t count;
	} images;

	struct
	{
		struct
		{
			cranvk_execution_ctx_t executionCtx;
			VkFence waitFence;
		} promises[cranvk_max_promise_count];

		// We don't know when we'll request our promises to be complete.
		// Instead of our usual queue allocation as we don't really expect to release resources, we have a bool set for each.
		// TODO: If ever bools are too expensive. Move to a bit scheme.
		bool allocatedPromises[cranvk_max_promise_count];
	} commandPromises;

	VkDescriptorPool descriptorPool;
	VkPipelineCache pipelineCache;
	VkCommandPool graphicsCommandPool;
	VkFence immediateFence;

	cranvk_allocator_t allocator;
} cranvk_graphics_device_t;

typedef struct
{
	VkRenderPass renderPass;
	uint32_t backBufferIndex;
	uint32_t framebufferIndices[cranvk_render_buffer_count];
	VkCommandBuffer primaryBuffers[cranvk_render_buffer_count];

	VkSemaphore finishedSemaphores[cranvk_render_buffer_count];
	VkFence finishedFences[cranvk_render_buffer_count];
} cranvk_render_pass_t;

typedef struct
{
	VkSurfaceFormatKHR surfaceFormat;
	VkExtent2D surfaceExtents;
	VkPresentModeKHR presentMode;

	VkSemaphore acquireSemaphores[cranvk_render_buffer_count];

	struct
	{
		VkSwapchainKHR swapchain;
		VkImageView imageViews[cranvk_render_buffer_count];

		// Tells us the framebuffers that need to be recreated on window resize
		struct
		{
			uint32_t imageViewIndices[cranvk_max_framebuffer_count];
			uint32_t framebufferIndices[cranvk_max_framebuffer_count];
			uint32_t count;
		} allocatedFramebuffers;
	} swapchainData;

	cranvk_render_pass_t presentRenderPass;
} cranvk_present_t;

unsigned int crang_ctx_size(void)
{
	return sizeof(cranvk_ctx_t);
}

crang_ctx_t* crang_create_ctx(void* buffer)
{
	cranvk_ctx_t* vkCtx = (cranvk_ctx_t*)buffer;

	VkApplicationInfo appInfo =
	{
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pNext = NULL,
		.pApplicationName = "CranberryGfx",
		.applicationVersion = 1,
		.pEngineName = "CranberryGfx",
		.engineVersion = 1,
		.apiVersion = VK_MAKE_VERSION(1, 0, VK_HEADER_VERSION)
	};

	VkInstanceCreateInfo createInfo =
	{
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pNext = NULL,
		.pApplicationInfo = &appInfo,
		.enabledExtensionCount = cranvk_instance_extension_count,
		.ppEnabledExtensionNames = cranvk_instance_extensions,
		.enabledLayerCount = cranvk_validation_count,
		.ppEnabledLayerNames = cranvk_validation_layers
	};

	cranvk_check(vkCreateInstance(&createInfo, cranvk_no_allocator, &vkCtx->instance));
	return (crang_ctx_t*)vkCtx;
}

void crang_destroy_ctx(crang_ctx_t* ctx)
{
	cranvk_ctx_t* vkCtx = (cranvk_ctx_t*)ctx;
	vkDestroyInstance(vkCtx->instance, cranvk_no_allocator);
}

unsigned int crang_win32_surface_size(void)
{
	return sizeof(cranvk_surface_t);
}

crang_surface_t* crang_win32_create_surface(void* buffer, crang_ctx_t* ctx, void* hinstance, void* window)
{
	cranvk_ctx_t* vkCtx = (cranvk_ctx_t*)ctx;

	cranvk_surface_t* surface = (cranvk_surface_t*)buffer;
	VkWin32SurfaceCreateInfoKHR surfaceCreateInfo =
	{
		.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
		.hinstance = hinstance,
		.hwnd = window
	};

	cranvk_check(vkCreateWin32SurfaceKHR(vkCtx->instance, &surfaceCreateInfo, cranvk_no_allocator, &surface->surface));
	return (crang_surface_t*)surface;
}

void crang_win32_destroy_surface(crang_ctx_t* ctx, crang_surface_t* surface)
{
	cranvk_ctx_t* vkCtx = (cranvk_ctx_t*)ctx;
	cranvk_surface_t* vkSurface = (cranvk_surface_t*)surface;
	vkDestroySurfaceKHR(vkCtx->instance, vkSurface->surface, cranvk_no_allocator);
}

unsigned int crang_graphics_device_size(void)
{
	return sizeof(cranvk_graphics_device_t);
}

crang_graphics_device_t* crang_create_graphics_device(void* buffer, crang_ctx_t* ctx, crang_surface_t* surface)
{
	cranvk_ctx_t* vkCtx = (cranvk_ctx_t*)ctx;
	cranvk_surface_t* vkSurface = (cranvk_surface_t*)surface;

	cranvk_graphics_device_t* vkDevice = (cranvk_graphics_device_t*)buffer;
	memset(buffer, 0, sizeof(cranvk_graphics_device_t));

	uint32_t physicalDeviceCount;
	VkPhysicalDevice physicalDevices[cranvk_max_physical_device_count];

	uint32_t queuePropertyCounts[cranvk_max_physical_device_count];
	VkQueueFamilyProperties queueProperties[cranvk_max_physical_device_count][cranvk_max_physical_device_property_count];

	// Enumerate the physical device properties
	{
		cranvk_check(vkEnumeratePhysicalDevices(vkCtx->instance, &physicalDeviceCount, NULL));
		physicalDeviceCount = physicalDeviceCount <= cranvk_max_physical_device_count ? physicalDeviceCount : cranvk_max_physical_device_count;

		cranvk_check(vkEnumeratePhysicalDevices(vkCtx->instance, &physicalDeviceCount, physicalDevices));

		for (uint32_t deviceIndex = 0; deviceIndex < physicalDeviceCount; deviceIndex++)
		{
			vkGetPhysicalDeviceQueueFamilyProperties(physicalDevices[deviceIndex], &queuePropertyCounts[deviceIndex], NULL);
			cranvk_assert(queuePropertyCounts[deviceIndex] > 0 && queuePropertyCounts[deviceIndex] <= cranvk_max_physical_device_property_count);

			queuePropertyCounts[deviceIndex] = queuePropertyCounts[deviceIndex] <= cranvk_max_physical_device_property_count ? queuePropertyCounts[deviceIndex] : cranvk_max_physical_device_property_count;
			vkGetPhysicalDeviceQueueFamilyProperties(physicalDevices[deviceIndex], &queuePropertyCounts[deviceIndex], queueProperties[deviceIndex]);
		}
	}

	// Select the device
	uint32_t physicalDeviceIndex = UINT32_MAX;
	{
		for (uint32_t deviceIndex = 0; deviceIndex < physicalDeviceCount; deviceIndex++)
		{
			uint32_t graphicsQueue = UINT32_MAX;
			uint32_t presentQueue = UINT32_MAX;

			for (uint32_t propIndex = 0; propIndex < queuePropertyCounts[deviceIndex]; propIndex++)
			{
				if (queueProperties[deviceIndex][propIndex].queueCount == 0)
				{
					continue;
				}

				if (queueProperties[deviceIndex][propIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT)
				{
					graphicsQueue = propIndex;
					break;
				}
			}

			for (uint32_t propIndex = 0; propIndex < queuePropertyCounts[deviceIndex]; propIndex++)
			{
				if (queueProperties[deviceIndex][propIndex].queueCount == 0)
				{
					continue;
				}

				VkBool32 supportsPresent = VK_FALSE;
				cranvk_check(vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevices[deviceIndex], propIndex, vkSurface->surface, &supportsPresent));

				if (supportsPresent)
				{
					presentQueue = propIndex;
					break;
				}
			}

			// Did we find a device supporting both graphics and present.
			if (graphicsQueue != UINT32_MAX && presentQueue != UINT32_MAX)
			{
				vkDevice->queues.graphicsQueueIndex = graphicsQueue;
				vkDevice->queues.presentQueueIndex = presentQueue;
				physicalDeviceIndex = deviceIndex;
				break;
			}
		}

		cranvk_assert(physicalDeviceIndex != UINT32_MAX);
	}

	// Create the logical device
	{
		VkDeviceQueueCreateInfo queueCreateInfo[2] = { { 0 }, { 0 } };
		uint32_t queueCreateInfoCount = 0;

		// Priority of 1.0f for everything. TODO: Do we want this to be customizable?
		static const float graphicsQueuePriority = 1.0f;
		VkDeviceQueueCreateInfo createGraphicsQueue =
		{
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueCount = 1,
			.queueFamilyIndex = vkDevice->queues.graphicsQueueIndex,
			.pQueuePriorities = &graphicsQueuePriority
		};

		queueCreateInfo[queueCreateInfoCount] = createGraphicsQueue;
		queueCreateInfoCount++;

		if (vkDevice->queues.graphicsQueueIndex != vkDevice->queues.presentQueueIndex)
		{
			static const float presentQueuePriority = 1.0f;

			VkDeviceQueueCreateInfo createPresentQueue =
			{
				.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
				.queueCount = 1,
				.queueFamilyIndex = vkDevice->queues.presentQueueIndex,
				.pQueuePriorities = &presentQueuePriority
			};

			queueCreateInfo[queueCreateInfoCount] = createPresentQueue;
			queueCreateInfoCount++;
		}

		VkPhysicalDeviceFeatures physicalDeviceFeatures = { 0 };
		VkDeviceCreateInfo deviceCreateInfo =
		{
			.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.enabledExtensionCount = cranvk_device_extension_count,
			.queueCreateInfoCount = queueCreateInfoCount,
			.pQueueCreateInfos = queueCreateInfo,
			.pEnabledFeatures = &physicalDeviceFeatures,
			.ppEnabledExtensionNames = cranvk_device_extensions,
			.enabledLayerCount = cranvk_validation_count,
			.ppEnabledLayerNames = cranvk_validation_layers
		};

		vkDevice->devices.physicalDevice = physicalDevices[physicalDeviceIndex];
		cranvk_check(vkCreateDevice(physicalDevices[physicalDeviceIndex], &deviceCreateInfo, cranvk_no_allocator, &vkDevice->devices.logicalDevice));

		vkGetDeviceQueue(vkDevice->devices.logicalDevice, vkDevice->queues.graphicsQueueIndex, 0, &vkDevice->queues.graphicsQueue);
		vkGetDeviceQueue(vkDevice->devices.logicalDevice, vkDevice->queues.presentQueueIndex, 0, &vkDevice->queues.presentQueue);
	}

	// Create the descriptor pools
	{
		VkDescriptorPoolSize descriptorPoolSizes[3] =
		{
			{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,.descriptorCount = cranvk_max_uniform_buffer_count },
			{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount = cranvk_max_storage_buffer_count },
			{ .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = cranvk_max_image_sampler_count }
		};

		VkDescriptorPoolCreateInfo descriptorPoolCreate =
		{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
			.maxSets = cranvk_max_descriptor_set_count,
			.poolSizeCount = 3,
			.pPoolSizes = descriptorPoolSizes
		};

		cranvk_check(vkCreateDescriptorPool(vkDevice->devices.logicalDevice, &descriptorPoolCreate, cranvk_no_allocator, &vkDevice->descriptorPool));
	}

	{
		VkPipelineCacheCreateInfo pipelineCacheCreate =
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO
		};

		cranvk_check(vkCreatePipelineCache(vkDevice->devices.logicalDevice, &pipelineCacheCreate, cranvk_no_allocator, &vkDevice->pipelineCache));
	}

	{
		VkCommandPoolCreateInfo commandPoolCreateInfo =
		{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = vkDevice->queues.graphicsQueueIndex
		};

		cranvk_check(vkCreateCommandPool(vkDevice->devices.logicalDevice, &commandPoolCreateInfo, cranvk_no_allocator, &vkDevice->graphicsCommandPool));
	}


	// command promises
	{
		VkFenceCreateInfo fenceCreateInfo =
		{
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = 0
		};

		for (uint32_t i = 0; i < cranvk_max_promise_count; i++)
		{
			cranvk_check(vkCreateFence(vkDevice->devices.logicalDevice, &fenceCreateInfo, cranvk_no_allocator, &vkDevice->commandPromises.promises[i].waitFence));
		}
	}

	VkFenceCreateInfo fenceCreateInfo =
	{
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.flags = 0
	};
	cranvk_check(vkCreateFence(vkDevice->devices.logicalDevice, &fenceCreateInfo, cranvk_no_allocator, &vkDevice->immediateFence));

	cranvk_create_allocator(&vkDevice->allocator);
	return (crang_graphics_device_t*)vkDevice;
}

void crang_destroy_graphics_device(crang_ctx_t* ctx, crang_graphics_device_t* device)
{
	cranvk_unused(ctx);
	cranvk_graphics_device_t* vkDevice = (cranvk_graphics_device_t*)device;

	vkDeviceWaitIdle(vkDevice->devices.logicalDevice);

	for (uint32_t i = 0; i < vkDevice->shaders.shaderCount; i++)
	{
		vkDestroyShaderModule(vkDevice->devices.logicalDevice, vkDevice->shaders.shaders[i], cranvk_no_allocator);
	}

	for (uint32_t i = 0; i < vkDevice->shaderLayouts.layoutCount; i++)
	{
		vkDestroyDescriptorSetLayout(vkDevice->devices.logicalDevice, vkDevice->shaderLayouts.descriptorSetLayouts[i], cranvk_no_allocator);
	}

	for (uint32_t i = 0; i < vkDevice->buffers.bufferCount; i++)
	{
		vkDestroyBuffer(vkDevice->devices.logicalDevice, vkDevice->buffers.buffers[i], cranvk_no_allocator);
		cranvk_allocator_free(&vkDevice->allocator, vkDevice->buffers.allocations[i]);
	}

	for (uint32_t i = 0; i < vkDevice->pipelines.pipelineCount; i++)
	{
		vkDestroyPipeline(vkDevice->devices.logicalDevice, vkDevice->pipelines.pipelines[i], cranvk_no_allocator);
		vkDestroyPipelineLayout(vkDevice->devices.logicalDevice, vkDevice->pipelines.layouts[i], cranvk_no_allocator);
	}

	for (uint32_t i = 0; i < vkDevice->framebuffers.count; i++)
	{
		vkDestroyFramebuffer(vkDevice->devices.logicalDevice, vkDevice->framebuffers.framebuffers[i], cranvk_no_allocator);
	}

	for (uint32_t i = 0; i < vkDevice->images.count; i++)
	{
		vkDestroySampler(vkDevice->devices.logicalDevice, vkDevice->images.samplers[i], cranvk_no_allocator);
		vkDestroyImageView(vkDevice->devices.logicalDevice, vkDevice->images.imageViews[i], cranvk_no_allocator);
		vkDestroyImage(vkDevice->devices.logicalDevice, vkDevice->images.images[i], cranvk_no_allocator);
	}

	for (uint32_t i = 0; i < cranvk_max_promise_count; i++)
	{
		vkDestroyFence(vkDevice->devices.logicalDevice, vkDevice->commandPromises.promises[i].waitFence, cranvk_no_allocator);
	}

	vkDestroyFence(vkDevice->devices.logicalDevice, vkDevice->immediateFence, cranvk_no_allocator);
	cranvk_destroy_allocator(vkDevice->devices.logicalDevice, &vkDevice->allocator);
	vkDestroyDescriptorPool(vkDevice->devices.logicalDevice, vkDevice->descriptorPool, cranvk_no_allocator);
	vkDestroyCommandPool(vkDevice->devices.logicalDevice, vkDevice->graphicsCommandPool, cranvk_no_allocator);
	vkDestroyPipelineCache(vkDevice->devices.logicalDevice, vkDevice->pipelineCache, cranvk_no_allocator);
	vkDestroyDevice(vkDevice->devices.logicalDevice, cranvk_no_allocator);
}

static void cranvk_allocate_framebuffers_from_swapchain(cranvk_graphics_device_t* vkDevice, cranvk_present_t* vkPresent)
{
	for (uint32_t i = 0; i < cranvk_render_buffer_count; i++)
	{
		uint32_t nextSwapchainFramebuffer = vkPresent->swapchainData.allocatedFramebuffers.count++;

		// Create the framebuffer
		uint32_t framebufferIndex;
		{
			VkFramebufferCreateInfo framebufferCreate =
			{
				.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
				.attachmentCount = 1,
				.width = vkPresent->surfaceExtents.width,
				.height = vkPresent->surfaceExtents.height,
				.layers = 1,
				.renderPass = vkPresent->presentRenderPass.renderPass,
				.pAttachments = &vkPresent->swapchainData.imageViews[i]
			};

			framebufferIndex = vkDevice->framebuffers.count++;
			cranvk_check(vkCreateFramebuffer(vkDevice->devices.logicalDevice, &framebufferCreate, cranvk_no_allocator, &vkDevice->framebuffers.framebuffers[framebufferIndex]));
		}

		// Tell the swapchain a framebuffer was allocated
		{
			vkPresent->swapchainData.allocatedFramebuffers.framebufferIndices[nextSwapchainFramebuffer] = framebufferIndex;
			vkPresent->swapchainData.allocatedFramebuffers.imageViewIndices[nextSwapchainFramebuffer] = i;
		}

		vkPresent->presentRenderPass.framebufferIndices[i] = framebufferIndex;
	}
}

static void cranvk_create_render_pass(cranvk_render_pass_t* vkRenderPass, cranvk_graphics_device_t* vkDevice, cranvk_present_t* vkPresent)
{
	// TODO: We'll want to be able to define our render pass here.
	{
		VkAttachmentDescription colorAttachment =
		{
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			.format = vkPresent->surfaceFormat.format
		};

		VkAttachmentReference colorReference =
		{
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		};

		VkSubpassDescription subpass =
		{
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 1,
			.pColorAttachments = &colorReference
		};

		VkRenderPassCreateInfo createRenderPass =
		{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = 1,
			.subpassCount = 1,
			.dependencyCount = 0,
			.pDependencies = NULL,
			.pAttachments = &colorAttachment,
			.pSubpasses = &subpass
		};

		cranvk_check(vkCreateRenderPass(vkDevice->devices.logicalDevice, &createRenderPass, cranvk_no_allocator, &vkRenderPass->renderPass));
	}

	cranvk_allocate_framebuffers_from_swapchain(vkDevice, vkPresent);

	{
		VkSemaphoreCreateInfo semaphoreCreateInfo =
		{
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
		};

		for (uint32_t i = 0; i < cranvk_render_buffer_count; i++)
		{
			cranvk_check(vkCreateSemaphore(vkDevice->devices.logicalDevice, &semaphoreCreateInfo, cranvk_no_allocator, &vkRenderPass->finishedSemaphores[i]));
		}
	}

	{
		VkCommandBufferAllocateInfo commandBufferAllocateInfo =
		{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = cranvk_render_buffer_count,
			.commandPool = vkDevice->graphicsCommandPool
		};

		cranvk_check(vkAllocateCommandBuffers(vkDevice->devices.logicalDevice, &commandBufferAllocateInfo, vkRenderPass->primaryBuffers));
	}

	{
		VkFenceCreateInfo fenceCreateInfo =
		{
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT
		};

		for (uint32_t i = 0; i < cranvk_render_buffer_count; i++)
		{
			cranvk_check(vkCreateFence(vkDevice->devices.logicalDevice, &fenceCreateInfo, cranvk_no_allocator, &vkRenderPass->finishedFences[i]));
		}
	}

	vkRenderPass->backBufferIndex = 0;
}

static void cranvk_destroy_render_pass(cranvk_graphics_device_t* device, cranvk_render_pass_t* vkRenderPass)
{
	cranvk_graphics_device_t* vkDevice = (cranvk_graphics_device_t*)device;

	vkDestroyRenderPass(vkDevice->devices.logicalDevice, vkRenderPass->renderPass, cranvk_no_allocator);

	for (uint32_t i = 0; i < cranvk_render_buffer_count; i++)
	{
		vkDestroySemaphore(vkDevice->devices.logicalDevice, vkRenderPass->finishedSemaphores[i], cranvk_no_allocator);
	}

	for (uint32_t i = 0; i < cranvk_render_buffer_count; i++)
	{
		vkDestroyFence(vkDevice->devices.logicalDevice, vkRenderPass->finishedFences[i], cranvk_no_allocator);
	}
}

static void cranvk_create_swapchain(cranvk_graphics_device_t* vkDevice, cranvk_surface_t* vkSurface, cranvk_present_t* vkPresent, VkSwapchainKHR oldSwapchain)
{
	VkSwapchainCreateInfoKHR swapchainCreate =
	{
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.pNext = NULL,
		.minImageCount = cranvk_render_buffer_count,
		.imageArrayLayers = 1,
		.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
		.surface = vkSurface->surface,
		.imageFormat = vkPresent->surfaceFormat.format,
		.imageColorSpace = vkPresent->surfaceFormat.colorSpace,
		.imageExtent = vkPresent->surfaceExtents,
		.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
		.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		.presentMode = vkPresent->presentMode,
		.clipped = VK_TRUE,
		.oldSwapchain = oldSwapchain,
		.pQueueFamilyIndices = NULL,
		.queueFamilyIndexCount = 0,
	};

	uint32_t swapchainShareIndices[2];
	if (vkDevice->queues.graphicsQueueIndex != vkDevice->queues.presentQueueIndex)
	{
		swapchainShareIndices[0] = vkDevice->queues.graphicsQueueIndex;
		swapchainShareIndices[1] = vkDevice->queues.presentQueueIndex;

		swapchainCreate.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		swapchainCreate.queueFamilyIndexCount = 2;
		swapchainCreate.pQueueFamilyIndices = swapchainShareIndices;
	}
	else
	{
		swapchainCreate.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	}

	cranvk_check(vkCreateSwapchainKHR(vkDevice->devices.logicalDevice, &swapchainCreate, cranvk_no_allocator, &vkPresent->swapchainData.swapchain));

	if (oldSwapchain != VK_NULL_HANDLE)
	{
		vkDestroySwapchainKHR(vkDevice->devices.logicalDevice, oldSwapchain, cranvk_no_allocator);

		for (uint32_t i = 0; i < cranvk_render_buffer_count; i++)
		{
			vkDestroyImageView(vkDevice->devices.logicalDevice, vkPresent->swapchainData.imageViews[i], cranvk_no_allocator);
		}

		for (uint32_t i = 0; i < vkPresent->swapchainData.allocatedFramebuffers.count; i++)
		{
			uint32_t framebufferIndex = vkPresent->swapchainData.allocatedFramebuffers.framebufferIndices[i];
			vkDestroyFramebuffer(vkDevice->devices.logicalDevice, vkDevice->framebuffers.framebuffers[framebufferIndex], cranvk_no_allocator);
		}
	}

	uint32_t imageCount = 0;
	VkImage swapchainPhysicalImages[cranvk_max_physical_image_count];

	cranvk_check(vkGetSwapchainImagesKHR(vkDevice->devices.logicalDevice, vkPresent->swapchainData.swapchain, &imageCount, NULL));
	imageCount = imageCount < cranvk_max_physical_image_count ? imageCount : cranvk_max_physical_image_count;
	cranvk_check(vkGetSwapchainImagesKHR(vkDevice->devices.logicalDevice, vkPresent->swapchainData.swapchain, &imageCount, swapchainPhysicalImages));

	for (uint32_t i = 0; i < cranvk_render_buffer_count; i++)
	{
		VkImageViewCreateInfo imageViewCreate =
		{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,

			.components.r = VK_COMPONENT_SWIZZLE_R,
			.components.g = VK_COMPONENT_SWIZZLE_G,
			.components.b = VK_COMPONENT_SWIZZLE_B,
			.components.a = VK_COMPONENT_SWIZZLE_A,

			.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.subresourceRange.baseMipLevel = 0,
			.subresourceRange.levelCount = 1,
			.subresourceRange.baseArrayLayer = 0,
			.subresourceRange.layerCount = 1,
			.image = swapchainPhysicalImages[i],
			.format = vkPresent->surfaceFormat.format
		};

		cranvk_check(vkCreateImageView(vkDevice->devices.logicalDevice, &imageViewCreate, cranvk_no_allocator, &vkPresent->swapchainData.imageViews[i]));
	}

	// Recreate the framebuffers that we already had
	for (uint32_t i = 0; i < vkPresent->swapchainData.allocatedFramebuffers.count; i++)
	{
		uint32_t framebufferIndex = vkPresent->swapchainData.allocatedFramebuffers.framebufferIndices[i];
		uint32_t imageViewIndex = vkPresent->swapchainData.allocatedFramebuffers.imageViewIndices[i];

		cranvk_assert(imageViewIndex < cranvk_render_buffer_count);

		VkFramebufferCreateInfo framebufferCreate =
		{
			.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			.attachmentCount = 1,
			.width = vkPresent->surfaceExtents.width,
			.height = vkPresent->surfaceExtents.height,
			.layers = 1,
			.renderPass = vkPresent->presentRenderPass.renderPass,
			.pAttachments = &vkPresent->swapchainData.imageViews[imageViewIndex]
		};

		cranvk_check(vkCreateFramebuffer(vkDevice->devices.logicalDevice, &framebufferCreate, cranvk_no_allocator, &vkDevice->framebuffers.framebuffers[framebufferIndex]));
	}
}

unsigned int crang_present_size(void)
{
	return sizeof(cranvk_present_t);
}

crang_present_t* crang_create_present(void* buffer, crang_graphics_device_t* device, crang_surface_t* surface)
{
	cranvk_present_t* vkPresent = (cranvk_present_t*)buffer;
	memset(vkPresent, 0, sizeof(cranvk_present_t));

	cranvk_graphics_device_t* vkDevice = (cranvk_graphics_device_t*)device;
	cranvk_surface_t* vkSurface = (cranvk_surface_t*)surface;

	{
		VkSemaphoreCreateInfo semaphoreCreateInfo =
		{
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
		};

		for (uint32_t i = 0; i < cranvk_render_buffer_count; i++)
		{
			cranvk_check(vkCreateSemaphore(vkDevice->devices.logicalDevice, &semaphoreCreateInfo, cranvk_no_allocator, &vkPresent->acquireSemaphores[i]));
		}
	}

	{
		uint32_t surfaceFormatCount;
		VkSurfaceFormatKHR surfaceFormats[cranvk_max_physical_device_property_count];

		cranvk_check(vkGetPhysicalDeviceSurfaceFormatsKHR(vkDevice->devices.physicalDevice, vkSurface->surface, &surfaceFormatCount, NULL));
		cranvk_assert(surfaceFormatCount > 0);
		surfaceFormatCount = surfaceFormatCount < cranvk_max_physical_device_property_count ? surfaceFormatCount : cranvk_max_physical_device_property_count;
		cranvk_check(vkGetPhysicalDeviceSurfaceFormatsKHR(vkDevice->devices.physicalDevice, vkSurface->surface, &surfaceFormatCount, surfaceFormats));

		if (1 == surfaceFormatCount && VK_FORMAT_UNDEFINED == surfaceFormats[0].format)
		{
			vkPresent->surfaceFormat.format = VK_FORMAT_R8G8B8A8_UNORM;
			vkPresent->surfaceFormat.colorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
		}
		else
		{
			vkPresent->surfaceFormat = surfaceFormats[0];
			for (uint32_t i = 0; i < surfaceFormatCount; i++)
			{
				if (VK_FORMAT_R8G8B8A8_UNORM == surfaceFormats[i].format && VK_COLORSPACE_SRGB_NONLINEAR_KHR == surfaceFormats[i].colorSpace)
				{
					vkPresent->surfaceFormat = surfaceFormats[i];
					break;
				}
			}
		}
	}

	{
		VkSurfaceCapabilitiesKHR surfaceCapabilities;
		cranvk_check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vkDevice->devices.physicalDevice, vkSurface->surface, &surfaceCapabilities));

		cranvk_assert(surfaceCapabilities.currentExtent.width != UINT32_MAX);
		vkPresent->surfaceExtents = surfaceCapabilities.currentExtent;
	}

	vkPresent->presentMode = VK_PRESENT_MODE_FIFO_KHR;
	{
		uint32_t presentModeCount;
		VkPresentModeKHR presentModes[cranvk_max_physical_device_property_count];

		cranvk_check(vkGetPhysicalDeviceSurfacePresentModesKHR(vkDevice->devices.physicalDevice, vkSurface->surface, &presentModeCount, NULL));
		cranvk_assert(presentModeCount > 0);
		presentModeCount = presentModeCount < cranvk_max_physical_device_property_count ? presentModeCount : cranvk_max_physical_device_property_count;
		cranvk_check(vkGetPhysicalDeviceSurfacePresentModesKHR(vkDevice->devices.physicalDevice, vkSurface->surface, &presentModeCount, presentModes));

		for (uint32_t i = 0; i < presentModeCount; i++)
		{
			if (VK_PRESENT_MODE_MAILBOX_KHR == presentModes[i])
			{
				vkPresent->presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
				break;
			}
		}
	}

	cranvk_create_swapchain(vkDevice, vkSurface, vkPresent, VK_NULL_HANDLE);

	cranvk_create_render_pass(&vkPresent->presentRenderPass, vkDevice, vkPresent);
	return (crang_present_t*)vkPresent;
}

void crang_destroy_present(crang_graphics_device_t* device, crang_present_t* presentCtx)
{
	cranvk_graphics_device_t* vkDevice = (cranvk_graphics_device_t*)device;
	cranvk_present_t* vkPresent = (cranvk_present_t*)presentCtx;

	vkDeviceWaitIdle(vkDevice->devices.logicalDevice);

	cranvk_destroy_render_pass(vkDevice, &vkPresent->presentRenderPass);

	for (uint32_t i = 0; i < cranvk_render_buffer_count; i++)
	{
		vkDestroyImageView(vkDevice->devices.logicalDevice, vkPresent->swapchainData.imageViews[i], cranvk_no_allocator);
	}

	for (uint32_t i = 0; i < cranvk_render_buffer_count; i++)
	{
		vkDestroySemaphore(vkDevice->devices.logicalDevice, vkPresent->acquireSemaphores[i], cranvk_no_allocator);
	}

	vkDestroySwapchainKHR(vkDevice->devices.logicalDevice, vkPresent->swapchainData.swapchain, cranvk_no_allocator);
}

static void cranvk_resize_present(cranvk_graphics_device_t* vkDevice, cranvk_surface_t* vkSurface, cranvk_present_t* vkPresent)
{
	vkDeviceWaitIdle(vkDevice->devices.logicalDevice);

	VkSurfaceCapabilitiesKHR surfaceCapabilities;
	cranvk_check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vkDevice->devices.physicalDevice, vkSurface->surface, &surfaceCapabilities));

	cranvk_assert(surfaceCapabilities.currentExtent.width != UINT32_MAX);
	vkPresent->surfaceExtents = surfaceCapabilities.currentExtent;

	cranvk_create_swapchain(vkDevice, vkSurface, vkPresent, vkPresent->swapchainData.swapchain);
}

crang_shader_id_t crang_request_shader_id(crang_graphics_device_t* device, crang_shader_e type)
{
	cranvk_graphics_device_t* vkDevice = (cranvk_graphics_device_t*)device;

	cranvk_assert(vkDevice->shaders.shaderCount < cranvk_max_shader_count);

	uint32_t nextSlot = vkDevice->shaders.shaderCount;
	vkDevice->shaders.shaderCount++;

	vkDevice->shaders.types[nextSlot] = type;

	return (crang_shader_id_t){ .id = nextSlot };
}

crang_shader_layout_id_t crang_request_shader_layout_id(crang_graphics_device_t* device, crang_shader_flags_e supportedTypes)
{
	cranvk_graphics_device_t* vkDevice = (cranvk_graphics_device_t*)device;

	cranvk_assert(vkDevice->shaderLayouts.layoutCount < cranvk_max_shader_layout_count);

	uint32_t nextSlot = vkDevice->shaderLayouts.layoutCount;
	vkDevice->shaderLayouts.layoutCount++;

	vkDevice->shaderLayouts.supportedTypes[nextSlot] = supportedTypes;

	return (crang_shader_layout_id_t){ .id = nextSlot };
}

crang_shader_input_id_t crang_request_shader_input_id(crang_graphics_device_t* device)
{
	cranvk_graphics_device_t* vkDevice = (cranvk_graphics_device_t*)device;

	uint32_t nextSlot = vkDevice->shaders.descriptorSets.count;
	vkDevice->shaders.descriptorSets.count++;
	cranvk_assert(vkDevice->shaders.descriptorSets.count <= cranvk_max_descriptor_set_count);

	return (crang_shader_input_id_t) { .id = nextSlot };
}

crang_buffer_id_t crang_request_buffer_id(crang_graphics_device_t* device)
{
	cranvk_graphics_device_t* vkDevice = (cranvk_graphics_device_t*)device;

	cranvk_assert(vkDevice->buffers.bufferCount < cranvk_max_buffer_count);

	uint32_t nextSlot = vkDevice->buffers.bufferCount;
	vkDevice->buffers.bufferCount++;

	return (crang_buffer_id_t){ .id = nextSlot };
}

crang_recording_buffer_id_t crang_request_recording_buffer_id(crang_graphics_device_t* device)
{
	cranvk_graphics_device_t* vkDevice = (cranvk_graphics_device_t*)device;

	cranvk_assert(vkDevice->commandBuffers.bufferCount < cranvk_max_command_buffer_count);

	uint32_t nextSlot = vkDevice->commandBuffers.bufferCount;
	vkDevice->commandBuffers.bufferCount++;

	VkCommandBufferAllocateInfo commandBufferAllocateInfo =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY,
		.commandBufferCount = 1,
		.commandPool = vkDevice->graphicsCommandPool
	};
	cranvk_check(vkAllocateCommandBuffers(vkDevice->devices.logicalDevice, &commandBufferAllocateInfo, &vkDevice->commandBuffers.recordingBuffers[nextSlot]));

	return (crang_recording_buffer_id_t){ .id = nextSlot };
}

crang_image_id_t crang_request_image_id(crang_graphics_device_t* device)
{
	cranvk_graphics_device_t* vkDevice = (cranvk_graphics_device_t*)device;
	cranvk_assert(vkDevice->images.count < cranvk_max_image_count);

	uint32_t nextSlot = vkDevice->images.count;
	vkDevice->images.count++;

	return (crang_image_id_t) { .id = nextSlot };
}

crang_pipeline_id_t crang_create_pipeline(crang_graphics_device_t* device, crang_pipeline_desc_t* pipelineDesc)
{
	cranvk_graphics_device_t* vkDevice = (cranvk_graphics_device_t*)device;

	cranvk_assert(vkDevice->pipelines.pipelineCount < cranvk_max_pipeline_count);
	crang_pipeline_id_t pipelineId = { .id = vkDevice->pipelines.pipelineCount };
	vkDevice->pipelines.pipelineCount++;


	cranvk_present_t* vkPresent = (cranvk_present_t*)pipelineDesc->presentCtx;
	cranvk_render_pass_t* vkRenderPass = &vkPresent->presentRenderPass;
	crang_shader_id_t vertShader = pipelineDesc->shaders[crang_shader_vertex];
	crang_shader_id_t fragShader = pipelineDesc->shaders[crang_shader_fragment];

	{
		VkDescriptorSetLayout descriptorSetLayouts[cranvk_max_pipeline_shader_layout_count];
		for (uint32_t i = 0; i < pipelineDesc->shaderLayouts.count; i++)
		{
			descriptorSetLayouts[i] = vkDevice->shaderLayouts.descriptorSetLayouts[pipelineDesc->shaderLayouts.layouts[i].id];
		}

		VkShaderStageFlagBits shaderStageConversionTable[] =
		{
			[crang_shader_flag_vertex] = VK_SHADER_STAGE_VERTEX_BIT,
			[crang_shader_flag_fragment] = VK_SHADER_STAGE_FRAGMENT_BIT,
			[crang_shader_flag_vertex | crang_shader_flag_fragment] = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
		};

		uint32_t immediateInputIndex = 0;
		VkPushConstantRange pushConstantRanges[cranvk_max_immediate_input_count];
		for (uint32_t i = 0; i < pipelineDesc->shaderLayouts.count; i++)
		{
			crang_shader_layout_id_t shaderLayout = pipelineDesc->shaderLayouts.layouts[i];
			crang_immediate_input_t immediateInput = vkDevice->shaderLayouts.immediateInputs[shaderLayout.id];


			if (immediateInput.size > 0)
			{
				// Make sure our push constant range is at minimum from 0 to 128. It's not ideal, but the spec states
				// that our push constant range has to be at least 128. (https://github.com/SaschaWillems/Vulkan/blob/ca17e359649c4f25ae9aa07352cfe5f532a8ca13/examples/pushconstants/pushconstants.cpp#L278)
				uint32_t offset;
				uint32_t size;
				{
					offset = immediateInput.offset;
					if (offset > 0)
					{
						offset = 0;
					}

					size = immediateInput.size;
					if (size < 128)
					{
						size = 128;
					}
				}

				pushConstantRanges[immediateInputIndex++] = (VkPushConstantRange)
				{
					.stageFlags = shaderStageConversionTable[vkDevice->shaderLayouts.supportedTypes[shaderLayout.id]],
					.offset = offset,
					.size = size
				};
			}
		}

		VkPipelineLayoutCreateInfo pipelineLayoutCreate =
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = pipelineDesc->shaderLayouts.count,
			.pSetLayouts = descriptorSetLayouts,
			.pushConstantRangeCount = immediateInputIndex,
			.pPushConstantRanges = pushConstantRanges
		};

		cranvk_check(vkCreatePipelineLayout(vkDevice->devices.logicalDevice, &pipelineLayoutCreate, cranvk_no_allocator, &vkDevice->pipelines.layouts[pipelineId.id]));
	}

	{
		VkVertexInputBindingDescription inputBindings[cranvk_max_vertex_inputs];
		for (uint32_t i = 0; i < pipelineDesc->vertexInputs.count; i++)
		{
			inputBindings[i] = (VkVertexInputBindingDescription)
			{
				.binding = pipelineDesc->vertexInputs.inputs[i].binding,
				.stride = pipelineDesc->vertexInputs.inputs[i].stride,
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX // TODO: We probably want a parameter for this
			};
		}

		VkFormat vkFormatConversionTable[crang_vertex_format_max] =
		{
			[crang_vertex_format_f32_1] = VK_FORMAT_R32_SFLOAT,
			[crang_vertex_format_f32_2] = VK_FORMAT_R32G32_SFLOAT,
			[crang_vertex_format_f32_3] = VK_FORMAT_R32G32B32_SFLOAT,
		};

		VkVertexInputAttributeDescription inputAttributes[cranvk_max_vertex_attributes];
		for (uint32_t i = 0; i < pipelineDesc->vertexAttributes.count; i++)
		{
			inputAttributes[i] = (VkVertexInputAttributeDescription)
			{
				.binding = pipelineDesc->vertexAttributes.attribs[i].binding,
				.location = pipelineDesc->vertexAttributes.attribs[i].location,
				.offset = pipelineDesc->vertexAttributes.attribs[i].offset,
				.format = vkFormatConversionTable[pipelineDesc->vertexAttributes.attribs[i].format]
			};
		}

		VkPipelineVertexInputStateCreateInfo vertexInputCreate =
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = pipelineDesc->vertexInputs.count,
			.pVertexBindingDescriptions = inputBindings,
			.vertexAttributeDescriptionCount = pipelineDesc->vertexAttributes.count,
			.pVertexAttributeDescriptions = inputAttributes
		};

		// Input Assembly
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyCreate =
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
		};

		// Rasterization
		VkPipelineRasterizationStateCreateInfo rasterizationCreate =
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.rasterizerDiscardEnable = VK_FALSE,
			.depthBiasEnable = VK_FALSE,
			.depthClampEnable = VK_FALSE,
			.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
			.lineWidth = 1.0f,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT
		};

		VkPipelineColorBlendAttachmentState colorBlendAttachment =
		{
			.blendEnable = VK_TRUE,
			.colorBlendOp = VK_BLEND_OP_ADD,
			.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
			.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
			.alphaBlendOp = VK_BLEND_OP_ADD,
			.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
			.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
		};

		VkPipelineColorBlendStateCreateInfo colorBlendCreate =
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.attachmentCount = 1,
			.pAttachments = &colorBlendAttachment
		};

		VkPipelineDepthStencilStateCreateInfo depthStencilCreate =
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = VK_FALSE,
			.depthWriteEnable = VK_FALSE,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
			.depthBoundsTestEnable = VK_FALSE,
			.minDepthBounds = 0.0f,
			.maxDepthBounds = 1.0f,
			.stencilTestEnable = VK_FALSE
			/*.front, .back */
		};

		VkPipelineMultisampleStateCreateInfo multisampleCreate =
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT
		};

		cranvk_assert(vkDevice->shaders.types[vertShader.id] == crang_shader_vertex);
		VkPipelineShaderStageCreateInfo vertexStage =
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pName = "main",
			.module = vkDevice->shaders.shaders[vertShader.id],
			.stage = VK_SHADER_STAGE_VERTEX_BIT
		};

		cranvk_assert(vkDevice->shaders.types[fragShader.id] == crang_shader_fragment);
		VkPipelineShaderStageCreateInfo fragStage =
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pName = "main",
			.module = vkDevice->shaders.shaders[fragShader.id],
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT
		};

		VkPipelineShaderStageCreateInfo shaderStages[crang_shader_max] = { vertexStage, fragStage };

		VkRect2D scissor =
		{
			{.x = 0,.y = 0 },
			.extent = vkPresent->surfaceExtents
		};

		// TODO: What about resizing?
		VkViewport viewport =
		{
			.x = 0,
			.y = 0,
			.width = (float)vkPresent->surfaceExtents.width,
			.height = (float)vkPresent->surfaceExtents.height,
			.minDepth = 0.0f,
			.maxDepth = 1.0f
		};

		VkPipelineViewportStateCreateInfo viewportStateCreate =
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.pViewports = &viewport,
			.scissorCount = 1,
			.pScissors = &scissor
		};

		VkGraphicsPipelineCreateInfo graphicsPipelineCreate =
		{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.layout = vkDevice->pipelines.layouts[pipelineId.id],
			.renderPass = vkRenderPass->renderPass,
			.pVertexInputState = &vertexInputCreate,
			.pInputAssemblyState = &inputAssemblyCreate,
			.pRasterizationState = &rasterizationCreate,
			.pColorBlendState = &colorBlendCreate,
			.pDepthStencilState = &depthStencilCreate,
			.pMultisampleState = &multisampleCreate,

			.pDynamicState = NULL,
			.pViewportState = &viewportStateCreate,
			.stageCount = crang_shader_max, // TODO: This isn't correct but it works for now
			.pStages = shaderStages
		};

		cranvk_check(vkCreateGraphicsPipelines(vkDevice->devices.logicalDevice, vkDevice->pipelineCache, 1, &graphicsPipelineCreate, cranvk_no_allocator, &vkDevice->pipelines.pipelines[pipelineId.id]));
	}

	return pipelineId;
}

void crang_present(crang_present_desc_t* presentDesc)
{
	cranvk_present_t* vkPresent = (cranvk_present_t*)presentDesc->presentCtx;
	cranvk_graphics_device_t* vkDevice = (cranvk_graphics_device_t*)presentDesc->graphicsDevice;
	cranvk_render_pass_t* vkRenderPass = &vkPresent->presentRenderPass;
	cranvk_surface_t* vkSurface = (cranvk_surface_t*)presentDesc->surface;

	// Start the frame
	uint32_t currentBackBuffer = vkRenderPass->backBufferIndex;

	cranvk_check(vkWaitForFences(vkDevice->devices.logicalDevice, 1, &vkRenderPass->finishedFences[currentBackBuffer], VK_TRUE, UINT64_MAX));
	cranvk_check(vkResetFences(vkDevice->devices.logicalDevice, 1, &vkRenderPass->finishedFences[currentBackBuffer]));

	uint32_t imageIndex = 0;
	VkResult result = vkAcquireNextImageKHR(vkDevice->devices.logicalDevice, vkPresent->swapchainData.swapchain, UINT64_MAX, vkPresent->acquireSemaphores[currentBackBuffer], VK_NULL_HANDLE, &imageIndex);
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
	{
		cranvk_resize_present(vkDevice, vkSurface, vkPresent);
		return;
	}

	VkCommandBuffer currentCommands = vkRenderPass->primaryBuffers[currentBackBuffer];
	{
		cranvk_check(vkResetCommandBuffer(currentCommands, 0));

		VkCommandBufferBeginInfo beginBufferInfo =
		{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
		};
		cranvk_check(vkBeginCommandBuffer(currentCommands, &beginBufferInfo));

		VkClearColorValue clearColor = { .float32 = { presentDesc->clearColor[0], presentDesc->clearColor[1], presentDesc->clearColor[2], 1.0f } };
		VkClearValue clearValue =
		{
			.color = clearColor
		};

		uint32_t framebufferIndex = vkPresent->swapchainData.allocatedFramebuffers.framebufferIndices[currentBackBuffer];
		VkRenderPassBeginInfo renderPassBeginInfo =
		{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = vkRenderPass->renderPass,
			.framebuffer = vkDevice->framebuffers.framebuffers[framebufferIndex],
			.renderArea = { .extent = vkPresent->surfaceExtents },
			.clearValueCount = 1,
			.pClearValues = &clearValue
		};
		vkCmdBeginRenderPass(currentCommands, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);

		for (uint32_t i = 0; i < presentDesc->recordedBuffers.count; i++)
		{
			VkCommandBuffer recordedBuffer = vkDevice->commandBuffers.recordingBuffers[i];
			vkCmdExecuteCommands(currentCommands, 1, &recordedBuffer);
		}

		vkCmdEndRenderPass(currentCommands);
		vkEndCommandBuffer(currentCommands);
	}

	VkSemaphore* acquire = &vkPresent->acquireSemaphores[currentBackBuffer];
	VkSemaphore* finished = &vkRenderPass->finishedSemaphores[currentBackBuffer];

	VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	VkSubmitInfo submitInfo =
	{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &currentCommands,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = acquire,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = finished,
		.pWaitDstStageMask = &dstStageMask
	};

	cranvk_check(vkQueueSubmit(vkDevice->queues.graphicsQueue, 1, &submitInfo, vkRenderPass->finishedFences[currentBackBuffer]));

	VkPresentInfoKHR presentInfo =
	{
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = finished,
		.swapchainCount = 1,
		.pSwapchains = &vkPresent->swapchainData.swapchain,
		.pImageIndices = &imageIndex
	};

	VkResult presentResult = vkQueuePresentKHR(vkDevice->queues.presentQueue, &presentInfo);
	if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR)
	{
		cranvk_resize_present(vkDevice, vkSurface, vkPresent);
	}

	vkRenderPass->backBufferIndex = (vkRenderPass->backBufferIndex + 1) % cranvk_render_buffer_count;
}

static void cranvk_create_shader(cranvk_graphics_device_t* vkDevice, cranvk_execution_ctx_t* ctx, void* commandData)
{
	cranvk_unused(ctx);

	crang_cmd_create_shader_t* createShaderData = (crang_cmd_create_shader_t*)commandData;

	VkShaderModuleCreateInfo createShader =
	{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.pCode = (const uint32_t*)createShaderData->source,
		.codeSize = createShaderData->sourceSize
	};

	VkShaderModule* shader = &vkDevice->shaders.shaders[createShaderData->shaderId.id];
	cranvk_check(vkCreateShaderModule(vkDevice->devices.logicalDevice, &createShader, cranvk_no_allocator, shader));
}

static void cranvk_create_shader_layout(cranvk_graphics_device_t* vkDevice, cranvk_execution_ctx_t* ctx, void* commandData)
{
	cranvk_unused(ctx);

	crang_cmd_create_shader_layout_t* createShaderLayoutData = (crang_cmd_create_shader_layout_t*)commandData;

	VkShaderStageFlagBits shaderStageConversionTable[] =
	{
		[crang_shader_flag_vertex] = VK_SHADER_STAGE_VERTEX_BIT,
		[crang_shader_flag_fragment] = VK_SHADER_STAGE_FRAGMENT_BIT,
		[crang_shader_flag_vertex | crang_shader_flag_fragment] = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
	};

	VkDescriptorType descriptorTypeConversionTable[] =
	{
		[crang_shader_input_type_uniform_buffer] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		[crang_shader_input_type_storage_buffer] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		[crang_shader_input_type_sampler] = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
	};

	VkDescriptorSetLayoutBinding layoutBindings[cranvk_max_shader_inputs];
	for (uint32_t i = 0; i < createShaderLayoutData->shaderInputs.count; i++)
	{
		layoutBindings[i] = (VkDescriptorSetLayoutBinding)
		{
			.stageFlags = shaderStageConversionTable[vkDevice->shaderLayouts.supportedTypes[createShaderLayoutData->shaderLayoutId.id]],
			.binding = createShaderLayoutData->shaderInputs.inputs[i].binding,
			.descriptorType = descriptorTypeConversionTable[createShaderLayoutData->shaderInputs.inputs[i].type],
			.descriptorCount = 1
		};
	}

	VkDescriptorSetLayoutCreateInfo createLayout =
	{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = createShaderLayoutData->shaderInputs.count,
		.pBindings = layoutBindings
	};

	VkDescriptorSetLayout* layout = &vkDevice->shaderLayouts.descriptorSetLayouts[createShaderLayoutData->shaderLayoutId.id];
	cranvk_check(vkCreateDescriptorSetLayout(vkDevice->devices.logicalDevice, &createLayout, cranvk_no_allocator, layout));

	vkDevice->shaderLayouts.immediateInputs[createShaderLayoutData->shaderLayoutId.id] = createShaderLayoutData->immediateInput;
}

static void cranvk_create_shader_input(cranvk_graphics_device_t* vkDevice, cranvk_execution_ctx_t* ctx, void* commandData)
{
	cranvk_unused(ctx);
	crang_cmd_create_shader_input_t* shaderInput = (crang_cmd_create_shader_input_t*)commandData;

	VkDescriptorSetAllocateInfo descriptorSetAlloc =
	{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vkDevice->descriptorPool,
		.descriptorSetCount = 1,
		.pSetLayouts = &vkDevice->shaderLayouts.descriptorSetLayouts[shaderInput->shaderLayoutId.id]
	};
	cranvk_check(vkAllocateDescriptorSets(
		vkDevice->devices.logicalDevice, &descriptorSetAlloc,
		&vkDevice->shaders.descriptorSets.sets[shaderInput->shaderInputId.id]));
}

static void cranvk_set_shader_input_data(cranvk_graphics_device_t* vkDevice, cranvk_execution_ctx_t* context, void* commandData)
{
	cranvk_unused(context);
	crang_cmd_set_shader_input_data_t* setInput = (crang_cmd_set_shader_input_data_t*)commandData;

	VkDescriptorType descriptorTypeConversionTable[] =
	{
		[crang_shader_input_type_uniform_buffer] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		[crang_shader_input_type_storage_buffer] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		[crang_shader_input_type_sampler] = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
	};

	switch (setInput->type)
	{
		case crang_shader_input_type_uniform_buffer:
		case crang_shader_input_type_storage_buffer:
		{
			VkDescriptorBufferInfo bufferInfo =
			{
				.buffer = vkDevice->buffers.buffers[setInput->uniformBuffer.bufferId.id],
				.offset = setInput->uniformBuffer.offset,
				.range = setInput->uniformBuffer.size
			};

			VkWriteDescriptorSet writeDescriptorSet =
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = vkDevice->shaders.descriptorSets.sets[setInput->shaderInputId.id],
				.descriptorType = descriptorTypeConversionTable[setInput->type],
				.dstBinding = setInput->binding,
				.descriptorCount = 1,
				.pBufferInfo = &bufferInfo
			};

			vkUpdateDescriptorSets(vkDevice->devices.logicalDevice, 1, &writeDescriptorSet, 0, NULL);
		}
		break;
		case crang_shader_input_type_sampler:
		{
			VkDescriptorImageInfo imageInfo =
			{
				.imageView = vkDevice->images.imageViews[setInput->sampler.imageId.id],
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				.sampler = vkDevice->images.samplers[setInput->sampler.imageId.id]
			};

			VkWriteDescriptorSet writeDescriptorSet =
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = vkDevice->shaders.descriptorSets.sets[setInput->shaderInputId.id],
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.dstBinding = setInput->binding,
				.descriptorCount = 1,
				.pImageInfo = &imageInfo
			};

			vkUpdateDescriptorSets(vkDevice->devices.logicalDevice, 1, &writeDescriptorSet, 0, NULL);
		}
		break;
	}
	
}


static void cranvk_create_buffer(cranvk_graphics_device_t* vkDevice, cranvk_execution_ctx_t* ctx, void* commandData)
{
	cranvk_unused(ctx);

	VkBufferUsageFlags bufferUsages[crang_buffer_max] =
	{
		[crang_buffer_vertex] = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		[crang_buffer_index] = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		[crang_buffer_shader_storage_input] = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		[crang_buffer_shader_uniform_input] = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	};

	crang_cmd_create_buffer_t* createBufferData = (crang_cmd_create_buffer_t*)commandData;

	VkBufferCreateInfo bufferCreate =
	{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = createBufferData->size,
		.usage = bufferUsages[createBufferData->type] | VK_BUFFER_USAGE_TRANSFER_DST_BIT // buffers created through create buffer can always be transfered to
	};

	VkBuffer* buffer = &vkDevice->buffers.buffers[createBufferData->bufferId.id];
	cranvk_check(vkCreateBuffer(vkDevice->devices.logicalDevice, &bufferCreate, cranvk_no_allocator, buffer));

	VkMemoryRequirements memoryRequirements;
	vkGetBufferMemoryRequirements(vkDevice->devices.logicalDevice, *buffer, &memoryRequirements);

	unsigned int preferredBits = 0;
	uint32_t memoryIndex = cranvk_find_memory_index(vkDevice->devices.physicalDevice, memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, preferredBits);

	cranvk_allocation_t* allocation = &vkDevice->buffers.allocations[createBufferData->bufferId.id];
	*allocation = cranvk_allocator_allocate(vkDevice->devices.logicalDevice, &vkDevice->allocator, memoryIndex, createBufferData->size, memoryRequirements.alignment);

	cranvk_check(vkBindBufferMemory(vkDevice->devices.logicalDevice, *buffer, allocation->memory, allocation->offset));
}

static void cranvk_copy_to_buffer(cranvk_graphics_device_t* vkDevice, cranvk_execution_ctx_t* context, void* commandData)
{
	crang_cmd_copy_to_buffer_t* copyToBufferData = (crang_cmd_copy_to_buffer_t*)commandData;

	VkBuffer srcBuffer;
	cranvk_allocation_t allocation;
	{
		VkBufferCreateInfo bufferCreate =
		{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = copyToBufferData->size,
			.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		};

		cranvk_check(vkCreateBuffer(vkDevice->devices.logicalDevice, &bufferCreate, cranvk_no_allocator, &srcBuffer));

		VkMemoryRequirements memoryRequirements;
		vkGetBufferMemoryRequirements(vkDevice->devices.logicalDevice, srcBuffer, &memoryRequirements);

		uint32_t memoryIndex = cranvk_find_memory_index(vkDevice->devices.physicalDevice, memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		allocation = cranvk_allocator_allocate(vkDevice->devices.logicalDevice, &vkDevice->allocator, memoryIndex, copyToBufferData->size, memoryRequirements.alignment);
		cranvk_check(vkBindBufferMemory(vkDevice->devices.logicalDevice, srcBuffer, allocation.memory, allocation.offset));

		{
			void* memory;
			unsigned int flags = 0;
			cranvk_check(vkMapMemory(vkDevice->devices.logicalDevice, allocation.memory, allocation.offset, copyToBufferData->size, flags, &memory));
			memcpy(memory, (uint8_t*)copyToBufferData->data, copyToBufferData->size);

			VkMappedMemoryRange mappedMemory = 
			{
				.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
				.memory = allocation.memory,
				.offset = allocation.offset,
				.size = copyToBufferData->size
			};
			vkFlushMappedMemoryRanges(vkDevice->devices.logicalDevice, 1, &mappedMemory);
			vkUnmapMemory(vkDevice->devices.logicalDevice, allocation.memory);
		}
	}

	VkBuffer dstBuffer = vkDevice->buffers.buffers[copyToBufferData->bufferId.id];

	VkBufferCopy copy = 
	{
		.srcOffset = 0,
		.dstOffset = copyToBufferData->offset,
		.size = copyToBufferData->size
	};
	vkCmdCopyBuffer(context->commandBuffer, srcBuffer, dstBuffer, 1, &copy);

	context->singleUseResources.buffers[context->singleUseResources.bufferCount] = srcBuffer;
	context->singleUseResources.bufferCount++;
	cranvk_assert(context->singleUseResources.bufferCount <= cranvk_max_single_use_resource_count);

	context->singleUseResources.allocations[context->singleUseResources.allocationCount] = allocation;
	context->singleUseResources.allocationCount++;
	cranvk_assert(context->singleUseResources.allocationCount <= cranvk_max_single_use_resource_count);
}

static void cranvk_create_image(cranvk_graphics_device_t* vkDevice, cranvk_execution_ctx_t* ctx, void* commandData)
{
	cranvk_unused(ctx);

	crang_cmd_create_image_t* createImageData = (crang_cmd_create_image_t*)commandData;
	uint32_t imageId = createImageData->imageId.id;

	VkFormat formats[crang_image_format_max] =
	{
		[crang_image_format_r8g8b8a8] = VK_FORMAT_R8G8B8A8_UNORM
	};

	uint32_t formatSize[crang_image_format_max] =
	{
		[crang_image_format_r8g8b8a8] = 4
	};

	// Image
	{
		VkImageCreateInfo imageCreate =
		{
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = formats[createImageData->format],
			.extent = {.width = createImageData->width,.height = createImageData->height,.depth = 1 },
			.mipLevels = 1,
			.arrayLayers = 1,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT ,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE
		};

		cranvk_check(vkCreateImage(vkDevice->devices.logicalDevice, &imageCreate, cranvk_no_allocator, &vkDevice->images.images[imageId]));
	}

	// Allocation
	{
		VkMemoryRequirements memoryRequirements;
		vkGetImageMemoryRequirements(vkDevice->devices.logicalDevice, vkDevice->images.images[imageId], &memoryRequirements);

		unsigned int preferredBits = 0;
		uint32_t memoryIndex = cranvk_find_memory_index(vkDevice->devices.physicalDevice, memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, preferredBits);

		cranvk_allocation_t* allocation = &vkDevice->images.allocations[imageId];
		*allocation = cranvk_allocator_allocate(vkDevice->devices.logicalDevice, &vkDevice->allocator, memoryIndex,
			createImageData->width * createImageData->height * formatSize[createImageData->format], memoryRequirements.alignment);

		cranvk_check(vkBindImageMemory(vkDevice->devices.logicalDevice, vkDevice->images.images[imageId], allocation->memory, allocation->offset));
	}

	// Image view
	{
		VkImageViewCreateInfo  imageCreateViewInfo =
		{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = vkDevice->images.images[imageId],
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = formats[createImageData->format],

			.components = {.r = VK_COMPONENT_SWIZZLE_R,.g = VK_COMPONENT_SWIZZLE_G,.b = VK_COMPONENT_SWIZZLE_B,.a = VK_COMPONENT_SWIZZLE_A },
			.subresourceRange =
			{
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.levelCount = 1,
				.layerCount = 1,
				.baseMipLevel = 0
			}
		};

		cranvk_check(vkCreateImageView(vkDevice->devices.logicalDevice, &imageCreateViewInfo, cranvk_no_allocator, &vkDevice->images.imageViews[imageId]));
	}

	// Sampler
	{
		VkSamplerCreateInfo samplerCreate = 
		{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_NEAREST,
			.minFilter = VK_FILTER_NEAREST,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.anisotropyEnable = VK_FALSE,
			.maxAnisotropy = 0,
			.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
			.compareEnable = VK_FALSE,
			.compareOp = VK_COMPARE_OP_ALWAYS,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
			.mipLodBias = 0.0f,
			.minLod = 0.0f,
			.maxLod = 0.0f
		};

		cranvk_check(vkCreateSampler(vkDevice->devices.logicalDevice, &samplerCreate, cranvk_no_allocator, &vkDevice->images.samplers[imageId]));
	}
}

static void cranvk_copy_to_image(cranvk_graphics_device_t* vkDevice, cranvk_execution_ctx_t* ctx, void* commandData)
{
	crang_cmd_copy_to_image_t* copyToImage = (crang_cmd_copy_to_image_t*)commandData;
	uint32_t imageId = copyToImage->imageId.id;

	VkBuffer srcBuffer;
	cranvk_allocation_t allocation;
	{
		uint32_t formatSize[crang_image_format_max] =
		{
			[crang_image_format_r8g8b8a8] = 4
		};

		uint32_t bufferSize = copyToImage->width * copyToImage->height * formatSize[copyToImage->format];

		VkBufferCreateInfo bufferCreate =
		{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = bufferSize,
			.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		};

		cranvk_check(vkCreateBuffer(vkDevice->devices.logicalDevice, &bufferCreate, cranvk_no_allocator, &srcBuffer));

		VkMemoryRequirements memoryRequirements;
		vkGetBufferMemoryRequirements(vkDevice->devices.logicalDevice, srcBuffer, &memoryRequirements);

		uint32_t memoryIndex = cranvk_find_memory_index(vkDevice->devices.physicalDevice, memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		allocation = cranvk_allocator_allocate(vkDevice->devices.logicalDevice, &vkDevice->allocator, memoryIndex, bufferSize, memoryRequirements.alignment);
		cranvk_check(vkBindBufferMemory(vkDevice->devices.logicalDevice, srcBuffer, allocation.memory, allocation.offset));

		{
			void* memory;
			unsigned int flags = 0;
			cranvk_check(vkMapMemory(vkDevice->devices.logicalDevice, allocation.memory, allocation.offset, bufferSize, flags, &memory));
			memcpy(memory, (uint8_t*)copyToImage->data + copyToImage->offset, bufferSize);

			VkMappedMemoryRange mappedMemory = 
			{
				.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
				.memory = allocation.memory,
				.offset = allocation.offset,
				.size = bufferSize
			};
			vkFlushMappedMemoryRanges(vkDevice->devices.logicalDevice, 1, &mappedMemory);
			vkUnmapMemory(vkDevice->devices.logicalDevice, allocation.memory);
		}
	}

	// Image barrier UNDEFINED -> OPTIMAL
	{
		VkAccessFlagBits sourceAccessMask = 0;
		VkAccessFlagBits dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;

		VkImageMemoryBarrier imageBarrier =
		{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = vkDevice->images.images[imageId],
			.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,.levelCount = 1, .baseMipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
			.srcAccessMask = sourceAccessMask,
			.dstAccessMask = dstAccessMask
		};

		vkCmdPipelineBarrier(ctx->commandBuffer, sourceStage, destinationStage, 0, 0, NULL, 0, NULL, 1, &imageBarrier);
	}

	// Image copy
	{
		VkBufferImageCopy copyRegion = 
		{
			.bufferOffset = 0,
			.bufferRowLength = 0,
			.bufferImageHeight = 0,
			.imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
			.imageOffset = {.x = copyToImage->offsetX, .y = copyToImage->offsetY, .z = 0},
			.imageExtent = {.width = copyToImage->width, .height = copyToImage->height, .depth = 1}
		};

		vkCmdCopyBufferToImage(ctx->commandBuffer, srcBuffer, vkDevice->images.images[imageId], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
	}

	// Image barrier OPTIMAL -> FRAGMEN_SHADER
	{

		VkAccessFlagBits sourceAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		VkAccessFlagBits dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

		VkImageMemoryBarrier imageBarrier =
		{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = vkDevice->images.images[imageId],
			.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,.levelCount = 1, .baseMipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
			.srcAccessMask = sourceAccessMask,
			.dstAccessMask = dstAccessMask
		};

		vkCmdPipelineBarrier(ctx->commandBuffer, sourceStage, destinationStage, 0, 0, NULL, 0, NULL, 1, &imageBarrier);
	}

	ctx->singleUseResources.buffers[ctx->singleUseResources.bufferCount] = srcBuffer;
	ctx->singleUseResources.bufferCount++;
	cranvk_assert(ctx->singleUseResources.bufferCount <= cranvk_max_single_use_resource_count);

	ctx->singleUseResources.allocations[ctx->singleUseResources.allocationCount] = allocation;
	ctx->singleUseResources.allocationCount++;
	cranvk_assert(ctx->singleUseResources.allocationCount <= cranvk_max_single_use_resource_count);
}

static void cranvk_execute_callback(cranvk_graphics_device_t* vkDevice, cranvk_execution_ctx_t* context, void* commandData)
{
	cranvk_unused(vkDevice);
	cranvk_unused(context);

	crang_cmd_callback_t* callbackCmd = (crang_cmd_callback_t*)commandData;
	callbackCmd->callback(callbackCmd->data);
}

static void cranvk_bind_pipeline(cranvk_graphics_device_t* vkDevice, cranvk_execution_ctx_t* context, void* commandData)
{
	crang_cmd_bind_pipeline_t* bindPipelineCmd = (crang_cmd_bind_pipeline_t*)commandData;
	vkCmdBindPipeline(context->commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vkDevice->pipelines.pipelines[bindPipelineCmd->pipelineId.id]);
}

static void cranvk_bind_vertex_inputs(cranvk_graphics_device_t* vkDevice, cranvk_execution_ctx_t* context, void* commandData)
{
	crang_cmd_bind_vertex_inputs_t* vertexInputs = (crang_cmd_bind_vertex_inputs_t*)commandData;
	for (uint32_t i = 0; i < vertexInputs->count; i++)
	{
		vkCmdBindVertexBuffers(
			context->commandBuffer, vertexInputs->bindings[i].binding, 1,
			&vkDevice->buffers.buffers[vertexInputs->bindings[i].bufferId.id], &(VkDeviceSize){ vertexInputs->bindings[i].offset });
	}
}

static void cranvk_bind_index_input(cranvk_graphics_device_t* vkDevice, cranvk_execution_ctx_t* context, void* commandData)
{
	VkIndexType indexTypeConversionTable[] =
	{
		[crang_index_type_u16] = VK_INDEX_TYPE_UINT16,
		[crang_index_type_u32] = VK_INDEX_TYPE_UINT32
	};

	crang_cmd_bind_index_input_t* indexInput = (crang_cmd_bind_index_input_t*)commandData;
	vkCmdBindIndexBuffer(
		context->commandBuffer, vkDevice->buffers.buffers[indexInput->bufferId.id],
		(VkDeviceSize) { indexInput->offset }, indexTypeConversionTable[indexInput->indexType]);
}

static void cranvk_bind_shader_input(cranvk_graphics_device_t* vkDevice, cranvk_execution_ctx_t* context, void* commandData)
{
	crang_cmd_bind_shader_input_t* shaderInput = (crang_cmd_bind_shader_input_t*)commandData;
	vkCmdBindDescriptorSets(
		context->commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vkDevice->pipelines.layouts[shaderInput->pipelineId.id],
		shaderInput->shaderLayoutIndex, 1, &vkDevice->shaders.descriptorSets.sets[shaderInput->shaderInputId.id], 0, VK_NULL_HANDLE);
}

static void cranvk_immediate_shader_input(cranvk_graphics_device_t* vkDevice, cranvk_execution_ctx_t* context, void* commandData)
{
	VkShaderStageFlagBits shaderStageConversionTable[] =
	{
		[crang_shader_flag_vertex] = VK_SHADER_STAGE_VERTEX_BIT,
		[crang_shader_flag_fragment] = VK_SHADER_STAGE_FRAGMENT_BIT,
		[crang_shader_flag_vertex | crang_shader_flag_fragment] = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
	};

	crang_cmd_immediate_shader_input_t* immediateInput = (crang_cmd_immediate_shader_input_t*)commandData;
	vkCmdPushConstants(context->commandBuffer, vkDevice->pipelines.layouts[immediateInput->pipelineId.id],
		shaderStageConversionTable[immediateInput->supportedShaders], immediateInput->offset, immediateInput->size, immediateInput->data);
}

static void cranvk_draw_indexed(cranvk_graphics_device_t* vkDevice, cranvk_execution_ctx_t* context, void* commandData)
{
	cranvk_unused(vkDevice);
	crang_cmd_draw_indexed_t* drawIndexed = (crang_cmd_draw_indexed_t*)commandData;
	vkCmdDrawIndexed(context->commandBuffer, drawIndexed->indexCount, drawIndexed->instanceCount, drawIndexed->indexOffset, drawIndexed->vertexOffset, 0);
}

typedef void(*cranvk_cmd_processor)(cranvk_graphics_device_t*, cranvk_execution_ctx_t*, void*);
cranvk_cmd_processor cmdProcessors[] =
{
	[crang_cmd_create_shader] = &cranvk_create_shader,
	[crang_cmd_create_shader_layout] = &cranvk_create_shader_layout,
	[crang_cmd_create_shader_input] = &cranvk_create_shader_input,
	[crang_cmd_set_shader_input_data] = &cranvk_set_shader_input_data,
	[crang_cmd_create_buffer] = &cranvk_create_buffer,
	[crang_cmd_copy_to_buffer] = &cranvk_copy_to_buffer,
	[crang_cmd_create_image] = &cranvk_create_image,
	[crang_cmd_copy_to_image] = &cranvk_copy_to_image,
	[crang_cmd_callback] = &cranvk_execute_callback,
	[crang_cmd_bind_pipeline] = &cranvk_bind_pipeline,
	[crang_cmd_bind_vertex_inputs] = &cranvk_bind_vertex_inputs,
	[crang_cmd_bind_index_input] = &cranvk_bind_index_input,
	[crang_cmd_bind_shader_input] = &cranvk_bind_shader_input,
	[crang_cmd_immediate_shader_input] = &cranvk_immediate_shader_input,
	[crang_cmd_draw_indexed] = &cranvk_draw_indexed,
};

void crang_execute_commands_immediate(crang_graphics_device_t* device, crang_cmd_buffer_t* cmdBuffer)
{
	cranvk_graphics_device_t* vkDevice = (cranvk_graphics_device_t*)device;
	cranvk_execution_ctx_t context = { 0 };

	VkCommandBufferAllocateInfo commandBufferAllocateInfo =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
		.commandPool = vkDevice->graphicsCommandPool
	};
	cranvk_check(vkAllocateCommandBuffers(vkDevice->devices.logicalDevice, &commandBufferAllocateInfo, &context.commandBuffer));

	VkCommandBufferBeginInfo beginBufferInfo =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
	};
	cranvk_check(vkBeginCommandBuffer(context.commandBuffer, &beginBufferInfo));

	for (uint32_t i = 0; i < cmdBuffer->count; i++)
	{
		crang_cmd_e command = cmdBuffer->commandDescs[i];
		cmdProcessors[command](vkDevice, &context, cmdBuffer->commandDatas[i]);
	}

	cranvk_check(vkEndCommandBuffer(context.commandBuffer));
	VkSubmitInfo submitInfo =
	{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &context.commandBuffer
	};
	cranvk_check(vkQueueSubmit(vkDevice->queues.graphicsQueue, 1, &submitInfo, vkDevice->immediateFence));
	cranvk_check(vkWaitForFences(vkDevice->devices.logicalDevice, 1, &vkDevice->immediateFence, VK_TRUE, UINT64_MAX));
	cranvk_check(vkResetFences(vkDevice->devices.logicalDevice, 1, &vkDevice->immediateFence));

	vkFreeCommandBuffers(vkDevice->devices.logicalDevice, vkDevice->graphicsCommandPool, 1, &context.commandBuffer);

	for(uint32_t i = 0; i < context.singleUseResources.bufferCount; i++)
	{
		vkDestroyBuffer(vkDevice->devices.logicalDevice, context.singleUseResources.buffers[i], cranvk_no_allocator);
	}

	for(uint32_t i = 0; i < context.singleUseResources.allocationCount; i++)
	{
		cranvk_allocator_free(&vkDevice->allocator, context.singleUseResources.allocations[i]);
	}
}

crang_promise_id_t crang_execute_commands_async(crang_graphics_device_t* device, crang_cmd_buffer_t* cmdBuffer)
{
	cranvk_graphics_device_t* vkDevice = (cranvk_graphics_device_t*)device;

	uint32_t allocIndex = 0;
	for (uint32_t i = 0; i < cranvk_max_promise_count; i++)
	{
		if (!vkDevice->commandPromises.allocatedPromises[i])
		{
			vkDevice->commandPromises.allocatedPromises[i] = true;
			allocIndex = i;
			break;
		}
	}

	cranvk_execution_ctx_t* context = &vkDevice->commandPromises.promises[allocIndex].executionCtx;
	VkCommandBufferAllocateInfo commandBufferAllocateInfo =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
		.commandPool = vkDevice->graphicsCommandPool
	};
	cranvk_check(vkAllocateCommandBuffers(vkDevice->devices.logicalDevice, &commandBufferAllocateInfo, &context->commandBuffer));

	VkCommandBufferBeginInfo beginBufferInfo =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
	};
	cranvk_check(vkBeginCommandBuffer(context->commandBuffer, &beginBufferInfo));

	for (uint32_t i = 0; i < cmdBuffer->count; i++)
	{
		crang_cmd_e command = cmdBuffer->commandDescs[i];
		cmdProcessors[command](vkDevice, context, cmdBuffer->commandDatas[i]);
	}

	cranvk_check(vkEndCommandBuffer(context->commandBuffer));
	VkSubmitInfo submitInfo =
	{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &context->commandBuffer
	};
	cranvk_check(vkQueueSubmit(vkDevice->queues.graphicsQueue, 1, &submitInfo, vkDevice->commandPromises.promises[allocIndex].waitFence));

	return (crang_promise_id_t) { .id = allocIndex };
}

void crang_wait_promise(crang_graphics_device_t* device, crang_promise_id_t promise)
{
	cranvk_graphics_device_t* vkDevice = (cranvk_graphics_device_t*)device;

	cranvk_assert(vkDevice->commandPromises.allocatedPromises[promise.id]);

	cranvk_check(vkWaitForFences(vkDevice->devices.logicalDevice, 1, &vkDevice->commandPromises.promises[promise.id].waitFence, VK_TRUE, UINT64_MAX));
	cranvk_check(vkResetFences(vkDevice->devices.logicalDevice, 1, &vkDevice->commandPromises.promises[promise.id].waitFence));

	cranvk_execution_ctx_t* context = &vkDevice->commandPromises.promises[promise.id].executionCtx;
	vkFreeCommandBuffers(vkDevice->devices.logicalDevice, vkDevice->graphicsCommandPool, 1, &context->commandBuffer);

	for(uint32_t i = 0; i < context->singleUseResources.bufferCount; i++)
	{
		vkDestroyBuffer(vkDevice->devices.logicalDevice, context->singleUseResources.buffers[i], cranvk_no_allocator);
	}
	context->singleUseResources.bufferCount = 0;

	for(uint32_t i = 0; i < context->singleUseResources.allocationCount; i++)
	{
		cranvk_allocator_free(&vkDevice->allocator, context->singleUseResources.allocations[i]);
	}
	context->singleUseResources.allocationCount = 0;

	vkDevice->commandPromises.allocatedPromises[promise.id] = false;
}

void crang_record_commands(crang_graphics_device_t* device, crang_present_t* present, crang_recording_buffer_id_t recordingBuffer, crang_cmd_buffer_t* cmdBuffer)
{
	cranvk_graphics_device_t* vkDevice = (cranvk_graphics_device_t*)device;
	cranvk_present_t* vkPresent = (cranvk_present_t*)present;

	cranvk_execution_ctx_t context = { 0 };
	context.commandBuffer = vkDevice->commandBuffers.recordingBuffers[recordingBuffer.id];
	
	VkCommandBufferInheritanceInfo inheritanceInfo =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
		.renderPass = vkPresent->presentRenderPass.renderPass,
		.subpass = 0, // TODO: subpass is always 0 for now
		.framebuffer = VK_NULL_HANDLE, //TODO: Do we want to be able to specify the framebuffer?
		.occlusionQueryEnable = VK_FALSE,
		.queryFlags = 0,
		.pipelineStatistics = 0
	};

	VkCommandBufferBeginInfo beginBufferInfo =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT | VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT, // TODO: right now simultaneous usage is fine, should see about later
		.pInheritanceInfo = &inheritanceInfo,
	};
	cranvk_check(vkBeginCommandBuffer(context.commandBuffer, &beginBufferInfo));

	for (uint32_t i = 0; i < cmdBuffer->count; i++)
	{
		crang_cmd_e command = cmdBuffer->commandDescs[i];
		cmdProcessors[command](vkDevice, &context, cmdBuffer->commandDatas[i]);
	}

	cranvk_check(vkEndCommandBuffer(context.commandBuffer));
	memcpy(&vkDevice->commandBuffers.singleUseResources[recordingBuffer.id], &context.singleUseResources, sizeof(cranvk_transient_resources_t));
}

#endif // CRANBERRY_GFX_BACKEND_IMPLEMENTATION
