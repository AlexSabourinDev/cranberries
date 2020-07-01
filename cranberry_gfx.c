#include "cranberry_gfx.h"

#pragma warning(disable : 4204 4221)

#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

#define VK_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#define crang_array_size(a) (sizeof(a)/sizeof(a[0]))

#define crang_no_allocator NULL

#define crang_debug_enabled

#ifdef crang_debug_enabled
void crang_check_impl(VkResult result)
{
	if(VK_SUCCESS != result)
	{
		__debugbreak();
	}
}
#define crang_check(call) \
	do \
	{ \
		crang_check_impl(call); \
	} while(0)

#define crang_assert(call) \
	do \
	{ \
		if (!(call)) \
		{ \
			__debugbreak(); \
		} \
	} while (0)

#define crang_error() \
	do \
	{ \
		__debugbreak(); \
	} while(0)
#else
#define crang_check(call) call
#define crang_assert(call)
#define crang_error()
#endif // cranvk_debug_enabled

// Allocator

#define cranvk_max_allocator_pools 10
#define cranvk_max_memory_blocks 1000
#define cranvk_allocator_pool_size (1024 * 1024 * 64)

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
			vkFreeMemory(device, allocator->memoryPools[i].memory, NULL);
		}
	}
}

static cranvk_allocation_t cranvk_allocator_allocate(VkDevice device, cranvk_allocator_t* allocator, uint32_t memoryTypeIndex, VkDeviceSize size, VkDeviceSize alignment)
{
	assert(memoryTypeIndex != UINT32_MAX);

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

				VkResult result = vkAllocateMemory(device, &memoryAllocInfo, NULL, &memoryPool->memory);
				assert(result == VK_SUCCESS);

				assert(allocator->freeBlockCount > 0);
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

	assert(foundPoolIndex != UINT32_MAX);

	VkDeviceSize allocationSize = size + (alignment - size % alignment);
		// Fun little trick to round to next nearest power of 2 from https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
	// Reduce number by 1, will handle powers of 2
	allocationSize--;
	// Set all the lower bits to get a full set bit pattern giving us Pow2 - 1
	allocationSize |= allocationSize >> 1;
	allocationSize |= allocationSize >> 2;
	allocationSize |= allocationSize >> 4;
	allocationSize |= allocationSize >> 8;
	allocationSize |= allocationSize >> 16;
	// Add 1 to push to pow of 2
	allocationSize++;


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
		assert(false);
		return (cranvk_allocation_t) { 0 };
	}

	while (iter->size > allocationSize && iter->size / 2 > allocationSize )
	{
		VkDeviceSize newBlockSize = iter->size / 2;

		iter->allocated = true;
		assert(allocator->freeBlockCount >= 2);

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

// Main Renderer

typedef enum
{
	queue_present,
	queue_graphics,
	queue_compute,
	queue_transfer,
	queue_count
} queue_e;

#define crang_max_mesh_count 1000
#define crang_max_image_count 100
#define crang_max_sampler_count 1000
#define crang_double_buffer_count 2
#define crang_max_material_instance_count 100
#define crang_max_physical_image_count 10
typedef struct
{
	VkInstance vkInstance;
	cranvk_allocator_t vkAllocator;

	struct
	{
		VkPhysicalDevice vkPhysicalDevice;
		VkDevice vkDevice;
		VkSurfaceKHR vkSurface;
		VkSurfaceFormatKHR vkSurfaceFormat;
		VkExtent2D vkMaxSurfaceExtents;
		VkExtent2D vkSurfaceExtents;
		VkPresentModeKHR vkPresentMode;
		VkDescriptorPool vkDescriptorPool;
		VkPipelineCache vkPipelineCache;
		VkSwapchainKHR vkSwapchain;
		VkImage vkSwapchainImages[crang_max_physical_image_count];
		uint32_t vkSwapchainImageCount;
		VkRenderPass vkRenderPass;

		struct
		{
			uint32_t index;
			VkQueue vkQueue;
			VkCommandPool vkCommandPool;
		} queues[queue_count];

		VkFence vkImmediateFence;

		struct
		{
			VkSemaphore vkAcquireSemaphore;
			VkSemaphore vkFinishedSemaphore;
			VkImageView vkSwapchainImageView;
			VkFramebuffer vkFramebuffer;
			VkCommandBuffer vkPrimaryCommandBuffer;
			VkFence vkFinishedFence;
		} doubleBuffer[crang_double_buffer_count];
		uint32_t activeDoubleBuffer;
	} present;

	struct
	{
		struct
		{
			uint32_t vertexOffset;
			uint32_t vertexSize;
			uint32_t indexOffset;
			uint32_t indexCount;
		} meshes[crang_max_mesh_count];
		uint32_t meshCount;
		uint32_t nextOffset;

		VkBuffer vkMeshDataBuffers;
		uint32_t allocationSize;
		cranvk_allocation_t allocation;
	} geometry;

	struct
	{
		struct
		{
			VkImage vkImage;
			VkImageView vkImageView;
			cranvk_allocation_t allocation;
		} images[crang_max_image_count];
		uint32_t imageCount;

		struct
		{
			VkSampler vkSampler;
		} samplers[crang_max_sampler_count];
		uint32_t samplerCount;
	} textures;

	struct
	{
		struct
		{
			VkShaderModule vkGbufferVShader;
			VkDescriptorSetLayout vkGbufferShaderLayout;

			VkShaderModule vkGbufferFShader;
			VkShaderModule vkGbufferComputeShader;
			VkDescriptorSetLayout vkGbufferComputeDescriptorLayout;
			VkPipelineLayout vkGbufferComputePipelineLayout;
			VkPipeline vkGbufferComputePipeline;

			VkPipelineLayout vkPipelineLayout;
			VkPipeline vkDeferredPipeline;

			struct
			{
				cranvk_allocation_t gbufferAllocation;
				VkImage vkGbufferImage;
				VkImageView vkGbufferImageView;
				VkDescriptorSet vkGbufferComputeDescriptor;
			} doubleBuffer[crang_double_buffer_count];

			struct
			{
				struct
				{
					VkDescriptorSet vkGbufferShaderDescriptor;
					VkBuffer vkGbufferFShaderData;
					cranvk_allocation_t allocation;

				} doubleBuffer[crang_double_buffer_count];
			} instances[crang_max_material_instance_count];
			uint32_t instanceCount;
		} deferred;
	} materials;
} context_t;

// TODO: Reference to Windows, if we want multiplatform we'll have to change this.
#ifdef crang_debug_enabled
#define crang_instance_extension_count 3
const char* crang_instance_extensions[crang_instance_extension_count] = { VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME, VK_EXT_DEBUG_REPORT_EXTENSION_NAME };
#else
#define crang_instance_extension_count 2
const char* crang_instance_extensions[crang_instance_extension_count] = { VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME };
#endif // crang_debug_enabled

#define crang_device_extension_count 1
const char* crang_device_extensions[crang_device_extension_count] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

#ifdef crang_debug_enabled
#define crang_validation_count 1
const char* crang_validation_layers[crang_validation_count] = { "VK_LAYER_LUNARG_standard_validation" };
#else
#define crang_validation_count 0
const char* crang_validation_layers[crang_validation_count] = {};
#endif

crang_context_t* crang_init(crang_init_desc_t* desc)
{
	context_t* ctx = malloc(sizeof(context_t));
	memset(ctx, 0, sizeof(context_t));

	{
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
			.enabledExtensionCount = crang_instance_extension_count,
			.ppEnabledExtensionNames = crang_instance_extensions,
			.enabledLayerCount = crang_validation_count,
			.ppEnabledLayerNames = crang_validation_layers
		};

		crang_check(vkCreateInstance(&createInfo, crang_no_allocator, &ctx->vkInstance));
	}

	{
		VkWin32SurfaceCreateInfoKHR surfaceCreateInfo =
		{
			.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
			.hinstance = desc->win32.hinstance,
			.hwnd = desc->win32.hwindow
		};

		crang_check(vkCreateWin32SurfaceKHR(ctx->vkInstance, &surfaceCreateInfo, crang_no_allocator, &ctx->present.vkSurface));
	}

	// Device
	{
#define crang_max_physical_device_count 10
#define crang_max_physical_device_property_count 100

		uint32_t physicalDeviceCount;
		VkPhysicalDevice physicalDevices[crang_max_physical_device_count];

		uint32_t queuePropertyCounts[crang_max_physical_device_count];
		VkQueueFamilyProperties queueProperties[crang_max_physical_device_count][crang_max_physical_device_property_count];

		// Enumerate the physical device properties
		{
			crang_check(vkEnumeratePhysicalDevices(ctx->vkInstance, &physicalDeviceCount, NULL));
			physicalDeviceCount = physicalDeviceCount <= crang_max_physical_device_count ? physicalDeviceCount : crang_max_physical_device_count;

			crang_check(vkEnumeratePhysicalDevices(ctx->vkInstance, &physicalDeviceCount, physicalDevices));
			for (uint32_t deviceIndex = 0; deviceIndex < physicalDeviceCount; deviceIndex++)
			{
				vkGetPhysicalDeviceQueueFamilyProperties(physicalDevices[deviceIndex], &queuePropertyCounts[deviceIndex], NULL);
				crang_assert(queuePropertyCounts[deviceIndex] > 0 && queuePropertyCounts[deviceIndex] <= crang_max_physical_device_property_count);

				queuePropertyCounts[deviceIndex] = queuePropertyCounts[deviceIndex] <= crang_max_physical_device_property_count ? queuePropertyCounts[deviceIndex] : crang_max_physical_device_property_count;
				vkGetPhysicalDeviceQueueFamilyProperties(physicalDevices[deviceIndex], &queuePropertyCounts[deviceIndex], queueProperties[deviceIndex]);
			}
		}

		// Select the device
		{
			uint32_t physicalDeviceIndex = UINT32_MAX;
			for (uint32_t deviceIndex = 0; deviceIndex < physicalDeviceCount; deviceIndex++)
			{
				uint32_t graphicsQueue = UINT32_MAX;
				uint32_t computeQueue = UINT32_MAX;
				uint32_t presentQueue = UINT32_MAX;
				uint32_t transferQueue = UINT32_MAX;

				// Find our graphics queue
				for (uint32_t propIndex = 0; propIndex < queuePropertyCounts[deviceIndex]; propIndex++)
				{
					if (queueProperties[deviceIndex][propIndex].queueCount == 0)
					{
						continue;
					}

					if (queueProperties[deviceIndex][propIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT)
					{
						graphicsQueue = propIndex;
					}

					if (queueProperties[deviceIndex][propIndex].queueFlags & VK_QUEUE_COMPUTE_BIT)
					{
						computeQueue = propIndex;
					}

					if (queueProperties[deviceIndex][propIndex].queueFlags & VK_QUEUE_TRANSFER_BIT)
					{
						transferQueue = propIndex;
					}
				}

				// Find our present queue
				for (uint32_t propIndex = 0; propIndex < queuePropertyCounts[deviceIndex]; propIndex++)
				{
					if (queueProperties[deviceIndex][propIndex].queueCount == 0)
					{
						continue;
					}

					VkBool32 supportsPresent = VK_FALSE;
					crang_check(vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevices[deviceIndex], propIndex, ctx->present.vkSurface, &supportsPresent));

					if (supportsPresent)
					{
						presentQueue = propIndex;
						break;
					}
				}

				// Did we find a device supporting both graphics, present and compute.
				if (graphicsQueue != UINT32_MAX && presentQueue != UINT32_MAX && computeQueue != UINT32_MAX)
				{
					ctx->present.queues[queue_graphics].index = graphicsQueue;
					ctx->present.queues[queue_compute].index = computeQueue;
					ctx->present.queues[queue_present].index = presentQueue;
					ctx->present.queues[queue_transfer].index = transferQueue;
					physicalDeviceIndex = deviceIndex;
					break;
				}
			}

			crang_assert(physicalDeviceIndex != UINT32_MAX);
			ctx->present.vkPhysicalDevice = physicalDevices[physicalDeviceIndex];
		}

		// Create the logical device
		{
			VkDeviceQueueCreateInfo queueCreateInfo[queue_count] = { 0 };
			uint32_t queueCreateInfoCount = 0;

			for (uint32_t i = 0; i < queue_count; i++)
			{
				// Have we already checked this index
				{
					bool queueFound = false;
					for (uint32_t j = 0; j < i; j++)
					{
						if (ctx->present.queues[j].index == ctx->present.queues[i].index)
						{
							queueFound = true;
						}
					}

					if (queueFound)
					{
						continue;
					}
				}

				static const float queuePriority = 1.0f;
				queueCreateInfo[queueCreateInfoCount++] = (VkDeviceQueueCreateInfo)
				{
					.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
						.queueCount = 1,
						.queueFamilyIndex = ctx->present.queues[i].index,
						.pQueuePriorities = &queuePriority
				};
			}

			VkPhysicalDeviceFeatures physicalDeviceFeatures = { .shaderStorageImageWriteWithoutFormat = VK_TRUE };
			VkDeviceCreateInfo deviceCreateInfo =
			{
				.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
				.enabledExtensionCount = crang_device_extension_count,
				.queueCreateInfoCount = queueCreateInfoCount,
				.pQueueCreateInfos = queueCreateInfo,
				.pEnabledFeatures = &physicalDeviceFeatures,
				.ppEnabledExtensionNames = crang_device_extensions,
				.enabledLayerCount = crang_validation_count,
				.ppEnabledLayerNames = crang_validation_layers
			};

			crang_check(vkCreateDevice(ctx->present.vkPhysicalDevice, &deviceCreateInfo, crang_no_allocator, &ctx->present.vkDevice));
			for (uint32_t i = 0; i < queue_count; i++)
			{
				vkGetDeviceQueue(ctx->present.vkDevice, ctx->present.queues[i].index, 0, &ctx->present.queues[i].vkQueue);
			}
		}
	}

	// Create the descriptor pools
	{
#define crang_max_uniform_buffer_count 1000
#define crang_max_storage_buffer_count 1000
#define crang_max_storage_image_count 100
#define crang_max_image_sampler_count 1000
#define crang_max_descriptor_set_count 1000

		VkDescriptorPoolSize descriptorPoolSizes[] =
		{
			{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,.descriptorCount = crang_max_uniform_buffer_count },
			{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount = crang_max_storage_buffer_count },
			{ .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,.descriptorCount = crang_max_storage_image_count },
			{ .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = crang_max_image_sampler_count }
		};

		VkDescriptorPoolCreateInfo descriptorPoolCreate =
		{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
			.maxSets = crang_max_descriptor_set_count,
			.poolSizeCount = crang_array_size(descriptorPoolSizes),
			.pPoolSizes = descriptorPoolSizes
		};

		crang_check(vkCreateDescriptorPool(ctx->present.vkDevice, &descriptorPoolCreate, crang_no_allocator, &ctx->present.vkDescriptorPool));
	}

	{
		VkPipelineCacheCreateInfo pipelineCacheCreate =
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO
		};

		crang_check(vkCreatePipelineCache(ctx->present.vkDevice, &pipelineCacheCreate, crang_no_allocator, &ctx->present.vkPipelineCache));
	}

	for (uint32_t i = 0; i < queue_count; i++)
	{
		VkCommandPoolCreateInfo commandPoolCreateInfo =
		{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = ctx->present.queues[i].index
		};

		crang_check(vkCreateCommandPool(ctx->present.vkDevice, &commandPoolCreateInfo, crang_no_allocator, &ctx->present.queues[i].vkCommandPool));
	}

	{
		VkFenceCreateInfo fenceCreateInfo =
		{
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
		};
		crang_check(vkCreateFence(ctx->present.vkDevice, &fenceCreateInfo, crang_no_allocator, &ctx->present.vkImmediateFence));
	}

	cranvk_create_allocator(&ctx->vkAllocator);

	{
		VkSemaphoreCreateInfo semaphoreCreateInfo =
		{
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
		};

		for (uint32_t i = 0; i < crang_double_buffer_count; i++)
		{
			crang_check(vkCreateSemaphore(ctx->present.vkDevice, &semaphoreCreateInfo, crang_no_allocator, &ctx->present.doubleBuffer[i].vkAcquireSemaphore));
		}
	}

	{
		uint32_t surfaceFormatCount;
		VkSurfaceFormatKHR surfaceFormats[crang_max_physical_device_property_count];

		crang_check(vkGetPhysicalDeviceSurfaceFormatsKHR(ctx->present.vkPhysicalDevice, ctx->present.vkSurface, &surfaceFormatCount, NULL));
		crang_assert(surfaceFormatCount > 0);
		surfaceFormatCount = surfaceFormatCount < crang_max_physical_device_property_count ? surfaceFormatCount : crang_max_physical_device_property_count;
		crang_check(vkGetPhysicalDeviceSurfaceFormatsKHR(ctx->present.vkPhysicalDevice, ctx->present.vkSurface, &surfaceFormatCount, surfaceFormats));

		if (1 == surfaceFormatCount && VK_FORMAT_UNDEFINED == surfaceFormats[0].format)
		{
			ctx->present.vkSurfaceFormat.format = VK_FORMAT_R8G8B8A8_UNORM;
			ctx->present.vkSurfaceFormat.colorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
		}
		else
		{
			ctx->present.vkSurfaceFormat = surfaceFormats[0];
			for (uint32_t i = 0; i < surfaceFormatCount; i++)
			{
				if (VK_FORMAT_R8G8B8A8_UNORM == surfaceFormats[i].format && VK_COLORSPACE_SRGB_NONLINEAR_KHR == surfaceFormats[i].colorSpace)
				{
					ctx->present.vkSurfaceFormat = surfaceFormats[i];
					break;
				}
			}
		}
	}

	{
		VkSurfaceCapabilitiesKHR surfaceCapabilities;
		crang_check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx->present.vkPhysicalDevice, ctx->present.vkSurface, &surfaceCapabilities));

		crang_assert(surfaceCapabilities.currentExtent.width != UINT32_MAX);
		ctx->present.vkMaxSurfaceExtents = surfaceCapabilities.maxImageExtent;
		ctx->present.vkSurfaceExtents = surfaceCapabilities.currentExtent;
	}

	ctx->present.vkPresentMode = VK_PRESENT_MODE_FIFO_KHR;
	{
		uint32_t presentModeCount;
		VkPresentModeKHR presentModes[crang_max_physical_device_property_count];

		crang_check(vkGetPhysicalDeviceSurfacePresentModesKHR(ctx->present.vkPhysicalDevice, ctx->present.vkSurface, &presentModeCount, NULL));
		crang_assert(presentModeCount > 0);
		presentModeCount = presentModeCount < crang_max_physical_device_property_count ? presentModeCount : crang_max_physical_device_property_count;
		crang_check(vkGetPhysicalDeviceSurfacePresentModesKHR(ctx->present.vkPhysicalDevice, ctx->present.vkSurface, &presentModeCount, presentModes));

		for (uint32_t i = 0; i < presentModeCount; i++)
		{
			if (VK_PRESENT_MODE_MAILBOX_KHR == presentModes[i])
			{
				ctx->present.vkPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
				break;
			}
		}
	}

	// Swapchain
	{
		VkSwapchainCreateInfoKHR swapchainCreate =
		{
			.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
			.pNext = NULL,
			.minImageCount = crang_double_buffer_count,
			.imageArrayLayers = 1,
			.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
			.surface = ctx->present.vkSurface,
			.imageFormat = ctx->present.vkSurfaceFormat.format,
			.imageColorSpace = ctx->present.vkSurfaceFormat.colorSpace,
			.imageExtent = ctx->present.vkMaxSurfaceExtents,
			.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
			.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			.presentMode = ctx->present.vkPresentMode,
			.clipped = VK_TRUE,
			.pQueueFamilyIndices = NULL,
			.queueFamilyIndexCount = 0,
		};

		uint32_t swapchainShareIndices[2];
		if (ctx->present.queues[queue_graphics].index != ctx->present.queues[queue_present].index)
		{
			swapchainShareIndices[0] = ctx->present.queues[queue_graphics].index;
			swapchainShareIndices[1] = ctx->present.queues[queue_present].index;

			swapchainCreate.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			swapchainCreate.queueFamilyIndexCount = 2;
			swapchainCreate.pQueueFamilyIndices = swapchainShareIndices;
		}
		else
		{
			swapchainCreate.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		crang_check(vkCreateSwapchainKHR(ctx->present.vkDevice, &swapchainCreate, crang_no_allocator, &ctx->present.vkSwapchain));

		crang_check(vkGetSwapchainImagesKHR(ctx->present.vkDevice, ctx->present.vkSwapchain, &ctx->present.vkSwapchainImageCount, NULL));
		ctx->present.vkSwapchainImageCount = ctx->present.vkSwapchainImageCount < crang_max_physical_image_count ? ctx->present.vkSwapchainImageCount : crang_max_physical_image_count;
		crang_check(vkGetSwapchainImagesKHR(ctx->present.vkDevice, ctx->present.vkSwapchain, &ctx->present.vkSwapchainImageCount, ctx->present.vkSwapchainImages));

		for (uint32_t i = 0; i < crang_double_buffer_count; i++)
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
				.image = ctx->present.vkSwapchainImages[i],
				.format = ctx->present.vkSurfaceFormat.format
			};

			crang_check(vkCreateImageView(ctx->present.vkDevice, &imageViewCreate, crang_no_allocator, &ctx->present.doubleBuffer[i].vkSwapchainImageView));
		}
	}

	{
		// TODO: We'll want to be able to define our render pass here.
		{
			VkAttachmentDescription attachments[] =
			{
				// Gbuffer
				{
					.samples = VK_SAMPLE_COUNT_1_BIT,
					.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
					.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
					.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
					.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
					.format = VK_FORMAT_R8G8B8A8_UNORM
				},
				// Output
				{
					.samples = VK_SAMPLE_COUNT_1_BIT,
					.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
					.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
					.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
					.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
					.format = ctx->present.vkSurfaceFormat.format
				}
			};

			// Gbuffer attachment
			VkAttachmentReference gbufferAttachment =
			{
				.attachment = 0,
				.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
			};

			VkSubpassDescription subpasses[] =
			{
				{
					.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
					.colorAttachmentCount = 1,
					.pColorAttachments = &gbufferAttachment
				}
			};

			VkRenderPassCreateInfo createRenderPass =
			{
				.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
				.attachmentCount = crang_array_size(attachments),
				.subpassCount = crang_array_size(subpasses),
				.dependencyCount = 0,
				.pAttachments = attachments,
				.pSubpasses = subpasses
			};

			crang_check(vkCreateRenderPass(ctx->present.vkDevice, &createRenderPass, crang_no_allocator, &ctx->present.vkRenderPass));
		}

		for (uint32_t i = 0; i < crang_double_buffer_count; i++)
		{
			// Create the gbuffer image
			{
				{
					VkImageCreateInfo imageCreate =
					{
						.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
						.imageType = VK_IMAGE_TYPE_2D,
						.format = VK_FORMAT_R8G8B8A8_UNORM,
						.extent = (VkExtent3D) { .width = ctx->present.vkMaxSurfaceExtents.width,.height = ctx->present.vkMaxSurfaceExtents.height,.depth = 1 },
						.mipLevels = 1,
						.arrayLayers = 1,
						.samples = VK_SAMPLE_COUNT_1_BIT,
						.tiling = VK_IMAGE_TILING_OPTIMAL,
						.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
						.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
						.sharingMode = VK_SHARING_MODE_EXCLUSIVE
					};

					crang_check(vkCreateImage(ctx->present.vkDevice, &imageCreate, crang_no_allocator, &ctx->materials.deferred.doubleBuffer[i].vkGbufferImage));
				}

				// Allocation
				{
					VkMemoryRequirements memoryRequirements;
					vkGetImageMemoryRequirements(ctx->present.vkDevice, ctx->materials.deferred.doubleBuffer[i].vkGbufferImage, &memoryRequirements);

					unsigned int preferredBits = 0;
					uint32_t memoryIndex = cranvk_find_memory_index(ctx->present.vkPhysicalDevice, memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, preferredBits);

					cranvk_allocation_t* allocation = &ctx->materials.deferred.doubleBuffer[i].gbufferAllocation;
					*allocation = cranvk_allocator_allocate(ctx->present.vkDevice, &ctx->vkAllocator, memoryIndex,
						ctx->present.vkMaxSurfaceExtents.width * ctx->present.vkMaxSurfaceExtents.height * 4, memoryRequirements.alignment);

					crang_check(vkBindImageMemory(ctx->present.vkDevice, ctx->materials.deferred.doubleBuffer[i].vkGbufferImage, allocation->memory, allocation->offset));
				}

				// Image view
				{
					VkImageViewCreateInfo  imageCreateViewInfo =
					{
						.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
						.image = ctx->materials.deferred.doubleBuffer[i].vkGbufferImage,
						.viewType = VK_IMAGE_VIEW_TYPE_2D,
						.format = VK_FORMAT_R8G8B8A8_UNORM,

						.components = {.r = VK_COMPONENT_SWIZZLE_R,.g = VK_COMPONENT_SWIZZLE_G,.b = VK_COMPONENT_SWIZZLE_B,.a = VK_COMPONENT_SWIZZLE_A },
						.subresourceRange =
						{
							.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
							.levelCount = 1,
							.layerCount = 1,
							.baseMipLevel = 0
						}
					};

					crang_check(vkCreateImageView(ctx->present.vkDevice, &imageCreateViewInfo, crang_no_allocator, &ctx->materials.deferred.doubleBuffer[i].vkGbufferImageView));
				}
			}

			// Create the framebuffer
			{
				VkImageView images[] = 
				{
					ctx->materials.deferred.doubleBuffer[i].vkGbufferImageView,
					ctx->present.doubleBuffer[i].vkSwapchainImageView
				};

				VkFramebufferCreateInfo framebufferCreate =
				{
					.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
					.attachmentCount = crang_array_size(images),
					.width = ctx->present.vkMaxSurfaceExtents.width,
					.height = ctx->present.vkMaxSurfaceExtents.height,
					.layers = 1,
					.renderPass = ctx->present.vkRenderPass,
					.pAttachments = images
				};

				crang_check(vkCreateFramebuffer(ctx->present.vkDevice, &framebufferCreate, crang_no_allocator, &ctx->present.doubleBuffer[i].vkFramebuffer));
			}
		}

		{
			VkSemaphoreCreateInfo semaphoreCreateInfo =
			{
				.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
			};

			for (uint32_t i = 0; i < crang_double_buffer_count; i++)
			{
				crang_check(vkCreateSemaphore(ctx->present.vkDevice, &semaphoreCreateInfo, crang_no_allocator, &ctx->present.doubleBuffer[i].vkFinishedSemaphore));
			}
		}

		for (uint32_t i = 0; i < crang_double_buffer_count; i++)
		{
			VkCommandBufferAllocateInfo commandBufferAllocateInfo =
			{
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
				.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
				.commandBufferCount = 1,
				.commandPool = ctx->present.queues[queue_graphics].vkCommandPool
			};

			crang_check(vkAllocateCommandBuffers(ctx->present.vkDevice, &commandBufferAllocateInfo, &ctx->present.doubleBuffer[i].vkPrimaryCommandBuffer));
		}

		{
			VkFenceCreateInfo fenceCreateInfo =
			{
				.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
				.flags = VK_FENCE_CREATE_SIGNALED_BIT
			};

			for (uint32_t i = 0; i < crang_double_buffer_count; i++)
			{
				crang_check(vkCreateFence(ctx->present.vkDevice, &fenceCreateInfo, crang_no_allocator, &ctx->present.doubleBuffer[i].vkFinishedFence));
			}
		}
	}

	// Deferred lit mat
	{
		VkShaderModuleCreateInfo createVShader =
		{
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.pCode = (const uint32_t*)desc->materials.deferred.gbufferVShader,
			.codeSize = desc->materials.deferred.gbufferVShaderSize
		};
		crang_check(vkCreateShaderModule(ctx->present.vkDevice, &createVShader, crang_no_allocator, &ctx->materials.deferred.vkGbufferVShader));

		VkShaderModuleCreateInfo createFShader =
		{
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.pCode = (const uint32_t*)desc->materials.deferred.gbufferFShader,
			.codeSize = desc->materials.deferred.gbufferFShaderSize
		};
		crang_check(vkCreateShaderModule(ctx->present.vkDevice, &createFShader, crang_no_allocator, &ctx->materials.deferred.vkGbufferFShader));

		VkShaderModuleCreateInfo createComputeShader =
		{
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.pCode = (const uint32_t*)desc->materials.deferred.gbufferComputeShader,
			.codeSize = desc->materials.deferred.gbufferComputeShaderSize
		};
		crang_check(vkCreateShaderModule(ctx->present.vkDevice, &createComputeShader, crang_no_allocator, &ctx->materials.deferred.vkGbufferComputeShader));

		// Compute pipeline
		{
			VkDescriptorSetLayoutBinding layoutBindings[] = 
			{
				{
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
				},
				{
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
				},
			};

			VkDescriptorSetLayoutCreateInfo createLayout =
			{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = crang_array_size(layoutBindings),
				.pBindings = layoutBindings
			};
			crang_check(vkCreateDescriptorSetLayout(ctx->present.vkDevice, &createLayout, crang_no_allocator, &ctx->materials.deferred.vkGbufferComputeDescriptorLayout));

			VkPipelineLayoutCreateInfo pipelineLayoutCreate =
			{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = 1,
				.pSetLayouts = &ctx->materials.deferred.vkGbufferComputeDescriptorLayout,
			};

			crang_check(vkCreatePipelineLayout(ctx->present.vkDevice, &pipelineLayoutCreate, crang_no_allocator, &ctx->materials.deferred.vkGbufferComputePipelineLayout));


			for (uint32_t i = 0; i < crang_double_buffer_count; i++)
			{
				VkDescriptorSetAllocateInfo descriptorSetAlloc =
				{
					.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
					.descriptorPool = ctx->present.vkDescriptorPool,
					.descriptorSetCount = 1,
					.pSetLayouts = &ctx->materials.deferred.vkGbufferComputeDescriptorLayout
				};
				crang_check(vkAllocateDescriptorSets(
					ctx->present.vkDevice, &descriptorSetAlloc,
					&ctx->materials.deferred.doubleBuffer[i].vkGbufferComputeDescriptor));

				VkDescriptorImageInfo gbufferInfo =
				{
					.imageView = ctx->materials.deferred.doubleBuffer[i].vkGbufferImageView,
					.imageLayout = VK_IMAGE_LAYOUT_GENERAL
				};

				VkWriteDescriptorSet computeSourceImageSet =
				{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = ctx->materials.deferred.doubleBuffer[i].vkGbufferComputeDescriptor,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
					.dstBinding = 0,
					.descriptorCount = 1,
					.pImageInfo = &gbufferInfo
				};

				VkDescriptorImageInfo swapchainInfo =
				{
					.imageView = ctx->present.doubleBuffer[i].vkSwapchainImageView,
					.imageLayout = VK_IMAGE_LAYOUT_GENERAL
				};

				VkWriteDescriptorSet swapchainImageSet =
				{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = ctx->materials.deferred.doubleBuffer[i].vkGbufferComputeDescriptor,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
					.dstBinding = 1,
					.descriptorCount = 1,
					.pImageInfo = &swapchainInfo
				};

				VkWriteDescriptorSet writeComputeSets[] = { computeSourceImageSet, swapchainImageSet };
				vkUpdateDescriptorSets(ctx->present.vkDevice, crang_array_size(writeComputeSets), writeComputeSets, 0, NULL);
			}

			VkComputePipelineCreateInfo info =
			{
				.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
				.stage = 
				{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
					.stage = VK_SHADER_STAGE_COMPUTE_BIT,
					.module = ctx->materials.deferred.vkGbufferComputeShader,
					.pName = "main"
				},
				.layout = ctx->materials.deferred.vkGbufferComputePipelineLayout
			};
			crang_check(vkCreateComputePipelines(ctx->present.vkDevice, ctx->present.vkPipelineCache, 1, &info, crang_no_allocator, &ctx->materials.deferred.vkGbufferComputePipeline));
		}

		// Graphics pipeline
		VkDescriptorSetLayoutBinding layoutBindings[] = 
		{
			{
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.descriptorCount = 1
			},
			{
				.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				.binding = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 1
			},
			{
				.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				.binding = 2,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1
			},
		};

		VkDescriptorSetLayoutCreateInfo createLayout =
		{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = crang_array_size(layoutBindings),
			.pBindings = layoutBindings
		};
		crang_check(vkCreateDescriptorSetLayout(ctx->present.vkDevice, &createLayout, crang_no_allocator, &ctx->materials.deferred.vkGbufferShaderLayout));

		{
			VkPushConstantRange pushConstantRange =
			{
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
				.offset = 0,
				.size = 128
			};

			VkPipelineLayoutCreateInfo pipelineLayoutCreate =
			{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = 1,
				.pSetLayouts = &ctx->materials.deferred.vkGbufferShaderLayout,
				.pushConstantRangeCount = 1,
				.pPushConstantRanges = &pushConstantRange
			};

			crang_check(vkCreatePipelineLayout(ctx->present.vkDevice, &pipelineLayoutCreate, crang_no_allocator, &ctx->materials.deferred.vkPipelineLayout));
		}

		{
			VkPipelineVertexInputStateCreateInfo vertexInputCreate =
			{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
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
			};

			VkPipelineMultisampleStateCreateInfo multisampleCreate =
			{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
				.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT
			};

			VkPipelineShaderStageCreateInfo vertexStage =
			{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.pName = "main",
				.module = ctx->materials.deferred.vkGbufferVShader,
				.stage = VK_SHADER_STAGE_VERTEX_BIT
			};

			VkPipelineShaderStageCreateInfo fragStage =
			{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.pName = "main",
				.module = ctx->materials.deferred.vkGbufferFShader,
				.stage = VK_SHADER_STAGE_FRAGMENT_BIT
			};

			VkPipelineShaderStageCreateInfo shaderStages[2] = { vertexStage, fragStage };
			VkRect2D scissor =
			{
				{.x = 0,.y = 0 },
				.extent = ctx->present.vkSurfaceExtents
			};

			// TODO: What about resizing?
			VkViewport viewport =
			{
				.x = 0,
				.y = 0,
				.width = (float)ctx->present.vkSurfaceExtents.width,
				.height = (float)ctx->present.vkSurfaceExtents.height,
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
				.layout = ctx->materials.deferred.vkPipelineLayout,
				.renderPass = ctx->present.vkRenderPass,
				.pVertexInputState = &vertexInputCreate,
				.pInputAssemblyState = &inputAssemblyCreate,
				.pRasterizationState = &rasterizationCreate,
				.pColorBlendState = &colorBlendCreate,
				.pDepthStencilState = &depthStencilCreate,
				.pMultisampleState = &multisampleCreate,

				.pDynamicState = NULL,
				.pViewportState = &viewportStateCreate,
				.stageCount = 2,
				.pStages = shaderStages
			};
			crang_check(vkCreateGraphicsPipelines(ctx->present.vkDevice, ctx->present.vkPipelineCache, 1, &graphicsPipelineCreate, crang_no_allocator, &ctx->materials.deferred.vkDeferredPipeline));
		}
	}

#define crang_mesh_buffer_size 1024*1024
	{
		VkBufferCreateInfo bufferCreate =
		{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = crang_mesh_buffer_size,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT // buffers created through create buffer can always be transfered to
		};

		VkBuffer* buffer = &ctx->geometry.vkMeshDataBuffers;
		ctx->geometry.allocationSize = crang_mesh_buffer_size;
		crang_check(vkCreateBuffer(ctx->present.vkDevice, &bufferCreate, crang_no_allocator, buffer));

		VkMemoryRequirements memoryRequirements;
		vkGetBufferMemoryRequirements(ctx->present.vkDevice, *buffer, &memoryRequirements);

		unsigned int preferredBits = 0;
		uint32_t memoryIndex = cranvk_find_memory_index(ctx->present.vkPhysicalDevice, memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, preferredBits);

		cranvk_allocation_t* allocation = &ctx->geometry.allocation;
		*allocation = cranvk_allocator_allocate(ctx->present.vkDevice, &ctx->vkAllocator, memoryIndex, crang_mesh_buffer_size, memoryRequirements.alignment);

		crang_check(vkBindBufferMemory(ctx->present.vkDevice, *buffer, allocation->memory, allocation->offset));
	}

	return (crang_context_t*)ctx;
}

crang_mesh_id_t crang_create_mesh(crang_context_t* context, crang_mesh_desc_t const* desc)
{
	context_t* ctx = (context_t*)context;

	uint32_t meshIndex = ctx->geometry.meshCount++;

	uint32_t vertexSize = desc->vertices.count * sizeof(crang_vertex_t);
	uint32_t indexSize = desc->indices.count * sizeof(uint32_t);
	{
		ctx->geometry.meshes[meshIndex].vertexSize = vertexSize;
		ctx->geometry.meshes[meshIndex].vertexOffset = ctx->geometry.nextOffset;
		ctx->geometry.meshes[meshIndex].indexOffset = vertexSize + ctx->geometry.nextOffset;
		ctx->geometry.meshes[meshIndex].indexCount = desc->indices.count;

		ctx->geometry.nextOffset += vertexSize + indexSize;
	}

	VkBuffer srcBuffer;
	cranvk_allocation_t allocation;
	{
		VkBufferCreateInfo bufferCreate =
		{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = indexSize + vertexSize,
			.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		};

		crang_check(vkCreateBuffer(ctx->present.vkDevice, &bufferCreate, crang_no_allocator, &srcBuffer));

		VkMemoryRequirements memoryRequirements;
		vkGetBufferMemoryRequirements(ctx->present.vkDevice, srcBuffer, &memoryRequirements);

		uint32_t memoryIndex = cranvk_find_memory_index(ctx->present.vkPhysicalDevice, memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		allocation = cranvk_allocator_allocate(ctx->present.vkDevice, &ctx->vkAllocator, memoryIndex, indexSize + vertexSize, memoryRequirements.alignment);
		crang_check(vkBindBufferMemory(ctx->present.vkDevice, srcBuffer, allocation.memory, allocation.offset));

		{
			void* memory;
			unsigned int flags = 0;
			crang_check(vkMapMemory(ctx->present.vkDevice, allocation.memory, allocation.offset, indexSize + vertexSize, flags, &memory));
			memcpy((uint8_t*)memory, (uint8_t*)desc->vertices.data, vertexSize);
			memcpy((uint8_t*)memory + vertexSize, (uint8_t*)desc->indices.data, indexSize);

			VkMappedMemoryRange mappedMemory = 
			{
				.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
				.memory = allocation.memory,
				.offset = allocation.offset,
				.size = indexSize + vertexSize
			};
			vkFlushMappedMemoryRanges(ctx->present.vkDevice, 1, &mappedMemory);
			vkUnmapMemory(ctx->present.vkDevice, allocation.memory);
		}
	}

	// TODO: We can likely store this in a cleanup structure when we request the dependency
	{
		VkCommandBufferAllocateInfo commandBufferAllocateInfo =
		{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
			.commandPool = ctx->present.queues[queue_transfer].vkCommandPool
		};
		VkCommandBuffer commandBuffer;
		crang_check(vkAllocateCommandBuffers(ctx->present.vkDevice, &commandBufferAllocateInfo, &commandBuffer));

		VkCommandBufferBeginInfo beginBufferInfo =
		{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
		};
		crang_check(vkBeginCommandBuffer(commandBuffer, &beginBufferInfo));

		VkBufferCopy meshCopy =
		{
			.srcOffset = 0,
			.dstOffset = ctx->geometry.meshes[meshIndex].vertexOffset,
			.size = vertexSize + indexSize
		};
		vkCmdCopyBuffer(commandBuffer, srcBuffer, ctx->geometry.vkMeshDataBuffers, 1, &meshCopy);

		crang_check(vkEndCommandBuffer(commandBuffer));
		VkSubmitInfo submitInfo =
		{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers = &commandBuffer
		};
		crang_check(vkQueueSubmit(ctx->present.queues[queue_transfer].vkQueue, 1, &submitInfo, ctx->present.vkImmediateFence));
		crang_check(vkWaitForFences(ctx->present.vkDevice, 1, &ctx->present.vkImmediateFence, VK_TRUE, UINT64_MAX));
		crang_check(vkResetFences(ctx->present.vkDevice, 1, &ctx->present.vkImmediateFence));

		vkFreeCommandBuffers(ctx->present.vkDevice,  ctx->present.queues[queue_transfer].vkCommandPool, 1, &commandBuffer);
		vkDestroyBuffer(ctx->present.vkDevice, srcBuffer, crang_no_allocator);
		cranvk_allocator_free(&ctx->vkAllocator, allocation);
	}

	return (crang_mesh_id_t) { meshIndex + 1 };
}

crang_image_id_t crang_create_image(crang_context_t* context, crang_image_desc_t const* desc)
{
	context_t* ctx = (context_t*)context;

	VkFormat formats[crang_image_format_count] =
	{
		[crang_image_format_rgba8] = VK_FORMAT_R8G8B8A8_UNORM
	};

	uint32_t formatSize[crang_image_format_count] =
	{
		[crang_image_format_rgba8] = 4
	};

	// Image
	uint32_t imageIndex = ctx->textures.imageCount++;
	{
		VkImageCreateInfo imageCreate =
		{
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = formats[desc->format],
			.extent = {.width = desc->width, .height = desc->height,.depth = 1 },
			.mipLevels = 1,
			.arrayLayers = 1,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT ,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE
		};

		crang_check(vkCreateImage(ctx->present.vkDevice, &imageCreate, crang_no_allocator, &ctx->textures.images[imageIndex].vkImage));
	}

	// Allocation
	{
		VkMemoryRequirements memoryRequirements;
		vkGetImageMemoryRequirements(ctx->present.vkDevice, ctx->textures.images[imageIndex].vkImage, &memoryRequirements);

		unsigned int preferredBits = 0;
		uint32_t memoryIndex = cranvk_find_memory_index(ctx->present.vkPhysicalDevice, memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, preferredBits);

		cranvk_allocation_t* allocation = &ctx->textures.images[imageIndex].allocation;
		*allocation = cranvk_allocator_allocate(ctx->present.vkDevice, &ctx->vkAllocator, memoryIndex,
			desc->width * desc->height * formatSize[desc->format], memoryRequirements.alignment);

		crang_check(vkBindImageMemory(ctx->present.vkDevice, ctx->textures.images[imageIndex].vkImage, allocation->memory, allocation->offset));
	}

	// Image view
	{
		VkImageViewCreateInfo  imageCreateViewInfo =
		{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = ctx->textures.images[imageIndex].vkImage,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = formats[desc->format],

			.components = {.r = VK_COMPONENT_SWIZZLE_R,.g = VK_COMPONENT_SWIZZLE_G,.b = VK_COMPONENT_SWIZZLE_B,.a = VK_COMPONENT_SWIZZLE_A },
			.subresourceRange =
			{
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.levelCount = 1,
				.layerCount = 1,
				.baseMipLevel = 0
			}
		};

		crang_check(vkCreateImageView(ctx->present.vkDevice, &imageCreateViewInfo, crang_no_allocator, &ctx->textures.images[imageIndex].vkImageView));
	}

	VkBuffer srcBuffer;
	cranvk_allocation_t allocation;
	{
		uint32_t bufferSize = desc->width * desc->height * formatSize[desc->format];
		VkBufferCreateInfo bufferCreate =
		{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = bufferSize,
			.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		};
		crang_check(vkCreateBuffer(ctx->present.vkDevice, &bufferCreate, crang_no_allocator, &srcBuffer));

		VkMemoryRequirements memoryRequirements;
		vkGetBufferMemoryRequirements(ctx->present.vkDevice, srcBuffer, &memoryRequirements);

		uint32_t memoryIndex = cranvk_find_memory_index(ctx->present.vkPhysicalDevice, memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		allocation = cranvk_allocator_allocate(ctx->present.vkDevice, &ctx->vkAllocator, memoryIndex, bufferSize, memoryRequirements.alignment);
		crang_check(vkBindBufferMemory(ctx->present.vkDevice, srcBuffer, allocation.memory, allocation.offset));

		{
			void* memory;
			unsigned int flags = 0;
			crang_check(vkMapMemory(ctx->present.vkDevice, allocation.memory, allocation.offset, bufferSize, flags, &memory));
			memcpy(memory, (uint8_t*)desc->data, bufferSize);

			VkMappedMemoryRange mappedMemory = 
			{
				.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
				.memory = allocation.memory,
				.offset = allocation.offset,
				.size = bufferSize
			};
			vkFlushMappedMemoryRanges(ctx->present.vkDevice, 1, &mappedMemory);
			vkUnmapMemory(ctx->present.vkDevice, allocation.memory);
		}
	}

	VkCommandBufferAllocateInfo commandBufferAllocateInfo =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
		.commandPool = ctx->present.queues[queue_graphics].vkCommandPool
	};
	VkCommandBuffer commandBuffer;
	crang_check(vkAllocateCommandBuffers(ctx->present.vkDevice, &commandBufferAllocateInfo, &commandBuffer));

	VkCommandBufferBeginInfo beginBufferInfo =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
	};
	crang_check(vkBeginCommandBuffer(commandBuffer, &beginBufferInfo));

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
			.image = ctx->textures.images[imageIndex].vkImage,
			.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,.levelCount = 1, .baseMipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
			.srcAccessMask = sourceAccessMask,
			.dstAccessMask = dstAccessMask
		};

		vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, NULL, 0, NULL, 1, &imageBarrier);
	}

	// Image copy
	{
		VkBufferImageCopy copyRegion = 
		{
			.bufferOffset = 0,
			.bufferRowLength = 0,
			.bufferImageHeight = 0,
			.imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
			.imageOffset = {.x = 0, .y = 0, .z = 0},
			.imageExtent = {.width = desc->width, .height = desc->height, .depth = 1}
		};

		vkCmdCopyBufferToImage(commandBuffer, srcBuffer, ctx->textures.images[imageIndex].vkImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
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
			.image = ctx->textures.images[imageIndex].vkImage,
			.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,.levelCount = 1, .baseMipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
			.srcAccessMask = sourceAccessMask,
			.dstAccessMask = dstAccessMask
		};

		vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, NULL, 0, NULL, 1, &imageBarrier);
	}

	crang_check(vkEndCommandBuffer(commandBuffer));
	VkSubmitInfo submitInfo =
	{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &commandBuffer
	};
	crang_check(vkQueueSubmit(ctx->present.queues[queue_graphics].vkQueue, 1, &submitInfo, ctx->present.vkImmediateFence));
	crang_check(vkWaitForFences(ctx->present.vkDevice, 1, &ctx->present.vkImmediateFence, VK_TRUE, UINT64_MAX));
	crang_check(vkResetFences(ctx->present.vkDevice, 1, &ctx->present.vkImmediateFence));

	vkFreeCommandBuffers(ctx->present.vkDevice,  ctx->present.queues[queue_graphics].vkCommandPool, 1, &commandBuffer);
	vkDestroyBuffer(ctx->present.vkDevice, srcBuffer, crang_no_allocator);
	cranvk_allocator_free(&ctx->vkAllocator, allocation);

	return (crang_image_id_t) { imageIndex + 1 };
}

crang_sampler_id_t crang_create_sampler(crang_context_t* context, crang_sampler_desc_t const* desc)
{
	(void)desc;
	context_t* ctx = (context_t*)context;

	// Sampler
	uint32_t samplerIndex = ctx->textures.samplerCount++;
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

		crang_check(vkCreateSampler(ctx->present.vkDevice, &samplerCreate, crang_no_allocator, &ctx->textures.samplers[samplerIndex].vkSampler));
	}

	return (crang_sampler_id_t) { samplerIndex + 1 };
}

crang_material_id_t crang_create_mat_deferred(crang_context_t* context, crang_deferred_desc_t const* desc)
{
	context_t* ctx = (context_t*)context;

	struct
	{
		float albedoTint[4];
	} matData;
	memcpy(matData.albedoTint, desc->albedoTint, sizeof(float) * 4);

	uint32_t instanceIndex = ctx->materials.deferred.instanceCount++;
	for (uint32_t i = 0; i < crang_double_buffer_count; i++)
	{
		VkBuffer dataBuffer;
		{
			VkBufferCreateInfo bufferCreate =
			{
				.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
				.size = sizeof(matData),
				.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT // buffers created through create buffer can always be transfered to
			};

			VkBuffer* buffer = &ctx->materials.deferred.instances[instanceIndex].doubleBuffer[i].vkGbufferFShaderData;
			crang_check(vkCreateBuffer(ctx->present.vkDevice, &bufferCreate, crang_no_allocator, buffer));

			VkMemoryRequirements memoryRequirements;
			vkGetBufferMemoryRequirements(ctx->present.vkDevice, *buffer, &memoryRequirements);

			unsigned int preferredBits = 0;
			uint32_t memoryIndex = cranvk_find_memory_index(ctx->present.vkPhysicalDevice, memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, preferredBits);

			cranvk_allocation_t* allocation = &ctx->materials.deferred.instances[instanceIndex].doubleBuffer[i].allocation;
			*allocation = cranvk_allocator_allocate(ctx->present.vkDevice, &ctx->vkAllocator, memoryIndex, sizeof(matData), memoryRequirements.alignment);

			crang_check(vkBindBufferMemory(ctx->present.vkDevice, *buffer, allocation->memory, allocation->offset));
			dataBuffer = *buffer;
		}

		VkBuffer srcBuffer;
		cranvk_allocation_t allocation;
		{
			VkBufferCreateInfo bufferCreate =
			{
				.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
				.size = sizeof(matData),
				.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
			};

			crang_check(vkCreateBuffer(ctx->present.vkDevice, &bufferCreate, crang_no_allocator, &srcBuffer));

			VkMemoryRequirements memoryRequirements;
			vkGetBufferMemoryRequirements(ctx->present.vkDevice, srcBuffer, &memoryRequirements);

			uint32_t memoryIndex = cranvk_find_memory_index(ctx->present.vkPhysicalDevice, memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

			allocation = cranvk_allocator_allocate(ctx->present.vkDevice, &ctx->vkAllocator, memoryIndex, sizeof(matData), memoryRequirements.alignment);
			crang_check(vkBindBufferMemory(ctx->present.vkDevice, srcBuffer, allocation.memory, allocation.offset));

			{
				void* memory;
				unsigned int flags = 0;
				crang_check(vkMapMemory(ctx->present.vkDevice, allocation.memory, allocation.offset, sizeof(matData), flags, &memory));
				memcpy((uint8_t*)memory, &matData, sizeof(matData));

				VkMappedMemoryRange mappedMemory = 
				{
					.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
					.memory = allocation.memory,
					.offset = allocation.offset,
					.size = sizeof(matData)
				};
				vkFlushMappedMemoryRanges(ctx->present.vkDevice, 1, &mappedMemory);
				vkUnmapMemory(ctx->present.vkDevice, allocation.memory);
			}
		}

		// TODO: We can likely store this in a cleanup structure when we request the dependency
		{
			VkCommandBufferAllocateInfo commandBufferAllocateInfo =
			{
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
				.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
				.commandBufferCount = 1,
				.commandPool = ctx->present.queues[queue_transfer].vkCommandPool
			};
			VkCommandBuffer commandBuffer;
			crang_check(vkAllocateCommandBuffers(ctx->present.vkDevice, &commandBufferAllocateInfo, &commandBuffer));

			VkCommandBufferBeginInfo beginBufferInfo =
			{
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
			};
			crang_check(vkBeginCommandBuffer(commandBuffer, &beginBufferInfo));

			VkBufferCopy copy =
			{
				.srcOffset = 0,
				.dstOffset = 0,
				.size = sizeof(matData)
			};
			vkCmdCopyBuffer(commandBuffer, srcBuffer, dataBuffer, 1, &copy);

			crang_check(vkEndCommandBuffer(commandBuffer));
			VkSubmitInfo submitInfo =
			{
				.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
				.commandBufferCount = 1,
				.pCommandBuffers = &commandBuffer
			};
			crang_check(vkQueueSubmit(ctx->present.queues[queue_transfer].vkQueue, 1, &submitInfo, ctx->present.vkImmediateFence));
			crang_check(vkWaitForFences(ctx->present.vkDevice, 1, &ctx->present.vkImmediateFence, VK_TRUE, UINT64_MAX));
			crang_check(vkResetFences(ctx->present.vkDevice, 1, &ctx->present.vkImmediateFence));

			vkFreeCommandBuffers(ctx->present.vkDevice, ctx->present.queues[queue_transfer].vkCommandPool, 1, &commandBuffer);
			vkDestroyBuffer(ctx->present.vkDevice, srcBuffer, crang_no_allocator);
			cranvk_allocator_free(&ctx->vkAllocator, allocation);
		}

		VkDescriptorSetAllocateInfo descriptorSetAlloc =
		{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = ctx->present.vkDescriptorPool,
			.descriptorSetCount = 1,
			.pSetLayouts = &ctx->materials.deferred.vkGbufferShaderLayout
		};
		crang_check(vkAllocateDescriptorSets(
			ctx->present.vkDevice, &descriptorSetAlloc,
			&ctx->materials.deferred.instances[instanceIndex].doubleBuffer[i].vkGbufferShaderDescriptor));
	
		VkDescriptorBufferInfo bufferInfo =
		{
			.buffer = dataBuffer,
			.offset = 0,
			.range = sizeof(matData)
		};

		VkWriteDescriptorSet writeMaterial =
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = ctx->materials.deferred.instances[instanceIndex].doubleBuffer[i].vkGbufferShaderDescriptor,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.dstBinding = 1,
			.descriptorCount = 1,
			.pBufferInfo = &bufferInfo
		};

		VkDescriptorBufferInfo meshBufferInfo =
		{
			.buffer = ctx->geometry.vkMeshDataBuffers,
			.offset = 0,
			.range = ctx->geometry.allocationSize
		};
		VkWriteDescriptorSet writeGeometry =
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = ctx->materials.deferred.instances[instanceIndex].doubleBuffer[i].vkGbufferShaderDescriptor,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.dstBinding = 0,
			.descriptorCount = 1,
			.pBufferInfo = &meshBufferInfo
		};

		VkDescriptorImageInfo imageInfo =
		{
			.imageView = ctx->textures.images[desc->albedoImage.id - 1].vkImageView,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.sampler = ctx->textures.samplers[desc->albedoSampler.id - 1].vkSampler
		};

		VkWriteDescriptorSet writeImageSet =
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = ctx->materials.deferred.instances[instanceIndex].doubleBuffer[i].vkGbufferShaderDescriptor,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.dstBinding = 2,
			.descriptorCount = 1,
			.pImageInfo = &imageInfo
		};

		VkWriteDescriptorSet writeDescriptorSets[] = { writeGeometry, writeMaterial, writeImageSet };
		vkUpdateDescriptorSets(ctx->present.vkDevice, crang_array_size(writeDescriptorSets), writeDescriptorSets, 0, NULL);
	}

	return (crang_material_id_t) { instanceIndex + 1 };
}

void crang_draw_view(crang_context_t* context, crang_view_t* view)
{
	context_t* ctx = (context_t*)context;

	// Start the frame
	uint32_t buffer = ctx->present.activeDoubleBuffer;

	uint32_t imageIndex = 0;
	VkResult result = vkAcquireNextImageKHR(ctx->present.vkDevice, ctx->present.vkSwapchain, UINT64_MAX, ctx->present.doubleBuffer[buffer].vkAcquireSemaphore, VK_NULL_HANDLE, &imageIndex);
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
	{
		vkDeviceWaitIdle(ctx->present.vkDevice);
		VkSurfaceCapabilitiesKHR surfaceCapabilities;
		crang_check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx->present.vkPhysicalDevice, ctx->present.vkSurface, &surfaceCapabilities));
		crang_assert(surfaceCapabilities.currentExtent.width != UINT32_MAX);
		ctx->present.vkSurfaceExtents = surfaceCapabilities.currentExtent;
		return;
	}

	crang_check(vkWaitForFences(ctx->present.vkDevice, 1, &ctx->present.doubleBuffer[ctx->present.activeDoubleBuffer].vkFinishedFence, VK_TRUE, UINT64_MAX));
	crang_check(vkResetFences(ctx->present.vkDevice, 1, &ctx->present.doubleBuffer[buffer].vkFinishedFence));

	VkCommandBuffer currentCommands = ctx->present.doubleBuffer[buffer].vkPrimaryCommandBuffer;
	{
		crang_check(vkResetCommandBuffer(currentCommands, 0));

		VkCommandBufferBeginInfo beginBufferInfo =
		{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = 0
		};
		crang_check(vkBeginCommandBuffer(currentCommands, &beginBufferInfo));

		VkClearColorValue clearColor = { .float32 = { 0.8f, 0.5f, 0.1f, 0.0f } };
		VkClearValue clearValue =
		{
			.color = clearColor
		};

		VkRenderPassBeginInfo renderPassBeginInfo =
		{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = ctx->present.vkRenderPass,
			.framebuffer = ctx->present.doubleBuffer[buffer].vkFramebuffer,
			.renderArea = {.extent = ctx->present.vkSurfaceExtents },
			.clearValueCount = 1,
			.pClearValues = &clearValue
		};
		vkCmdBeginRenderPass(currentCommands, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(currentCommands, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx->materials.deferred.vkDeferredPipeline);
		for (uint32_t i = 0; i < crang_max_batches && view->batches[crang_material_deferred][i].material.id > 0; i++)
		{
			uint32_t materialIndex = view->batches[crang_material_deferred][i].material.id - 1;
			VkDescriptorSet gbufferSet = ctx->materials.deferred.instances[materialIndex].doubleBuffer[buffer].vkGbufferShaderDescriptor;
			vkCmdBindDescriptorSets(
				currentCommands, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx->materials.deferred.vkPipelineLayout,
				0, 1, &gbufferSet, 0, VK_NULL_HANDLE);

			for (uint32_t inst = 0; inst < crang_max_instances && view->batches[crang_material_deferred][i].instances[inst].mesh.id > 0; inst++)
			{
				uint32_t meshIndex = view->batches[crang_material_deferred][i].instances[inst].mesh.id - 1;

				vkCmdBindIndexBuffer(currentCommands, ctx->geometry.vkMeshDataBuffers, ctx->geometry.meshes[meshIndex].indexOffset, VK_INDEX_TYPE_UINT32);
				for (uint32_t mat = 0; mat < view->batches[crang_material_deferred][i].instances[inst].count; mat++)
				{
					struct
					{
						crang_mat4_t vp;
						crang_mat4x3_t m;
						uint32_t vertexOffset;
					} pushConstant;

					pushConstant.vp = view->viewProj;
					pushConstant.m = view->batches[crang_material_deferred][i].instances[inst].transforms[mat];
					pushConstant.vertexOffset = ctx->geometry.meshes[meshIndex].vertexOffset / sizeof(crang_vertex_t);

					vkCmdPushConstants(currentCommands, ctx->materials.deferred.vkPipelineLayout,
						VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushConstant), &pushConstant);

					vkCmdDrawIndexed(currentCommands, ctx->geometry.meshes[meshIndex].indexCount, 1, 0, 0, 0);
				}
			}
		}

		vkCmdEndRenderPass(currentCommands);

		VkImageMemoryBarrier gbufferBarrier =
		{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_GENERAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = ctx->materials.deferred.doubleBuffer[buffer].vkGbufferImage,
			.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,.levelCount = 1, .baseMipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT
		};
		vkCmdPipelineBarrier(currentCommands, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &gbufferBarrier);

		VkImageMemoryBarrier swapchainBarrier =
		{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			.newLayout = VK_IMAGE_LAYOUT_GENERAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = ctx->present.vkSwapchainImages[imageIndex],
			.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,.levelCount = 1, .baseMipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT
		};
		vkCmdPipelineBarrier(currentCommands, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &swapchainBarrier);

		vkCmdBindPipeline(currentCommands, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->materials.deferred.vkGbufferComputePipeline);
		vkCmdBindDescriptorSets(currentCommands, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->materials.deferred.vkGbufferComputePipelineLayout, 0, 1,
								&ctx->materials.deferred.doubleBuffer[buffer].vkGbufferComputeDescriptor, 0, VK_NULL_HANDLE);
		vkCmdDispatch(currentCommands, ctx->present.vkSurfaceExtents.width, ctx->present.vkSurfaceExtents.height, 1);

		VkImageMemoryBarrier swapchainPresentBarrier =
		{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
			.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = ctx->present.vkSwapchainImages[imageIndex],
			.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,.levelCount = 1, .baseMipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
		};
		vkCmdPipelineBarrier(currentCommands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &swapchainPresentBarrier);

		vkEndCommandBuffer(currentCommands);
	}

	VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	VkSubmitInfo submitInfo =
	{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &currentCommands,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &ctx->present.doubleBuffer[buffer].vkAcquireSemaphore,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &ctx->present.doubleBuffer[buffer].vkFinishedSemaphore,
		.pWaitDstStageMask = &dstStageMask
	};

	crang_check(vkQueueSubmit(ctx->present.queues[queue_graphics].vkQueue, 1, &submitInfo, ctx->present.doubleBuffer[buffer].vkFinishedFence));

	VkPresentInfoKHR presentInfo =
	{
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &ctx->present.doubleBuffer[buffer].vkFinishedSemaphore,
		.swapchainCount = 1,
		.pSwapchains = &ctx->present.vkSwapchain,
		.pImageIndices = &imageIndex
	};

	VkResult presentResult = vkQueuePresentKHR(ctx->present.queues[queue_present].vkQueue, &presentInfo);
	if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR)
	{
		vkDeviceWaitIdle(ctx->present.vkDevice);
		VkSurfaceCapabilitiesKHR surfaceCapabilities;
		crang_check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx->present.vkPhysicalDevice, ctx->present.vkSurface, &surfaceCapabilities));
		crang_assert(surfaceCapabilities.currentExtent.width != UINT32_MAX);
		ctx->present.vkSurfaceExtents = surfaceCapabilities.currentExtent;
	}

	ctx->present.activeDoubleBuffer = (ctx->present.activeDoubleBuffer + 1) % crang_double_buffer_count;
}
