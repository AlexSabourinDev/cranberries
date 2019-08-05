#ifndef __CRANBERRY_PROCEDURAL_ASSETS_H
#define __CRANBERRY_PROCEDURAL_ASSETS_H

typedef enum
{
	cranpa_op_id_circle = 0,
	cranpa_op_id_translate,
	cranpa_op_id_max
} cranpa_op_id_e;

typedef struct _cranpa_vm_t cranpa_vm_t;

typedef void(*cranpa_opf_t)(cranpa_vm_t* vm, unsigned int slotId,  unsigned int* inputBuffers, unsigned int inputBufferCount, void* params);

// cranpa_script_t is a large memory buffer that is used by the virtual machine (or interpreter if you want to call it that)
// to actually run scripts. It's format looks similarly to [collectionSize, collection, collectionSize, collection]
// This means that we can't advance a constant amount, but our memory is fairly elegantly laid out.

// Currently, memory for ops in cranberry_procedural is simple. Every op gets the same amount of memory. This means that some ops will not
// make good use of their memory slot, but also allows allocations, deallocations and memory accesses to be fast. As a result, at the cost of
// more memory, the usability and performance are pretty good.

// NOTE: Meshes are always output with counter clockwise winding

//
// cranpa_script_t memory format:
// op count
// op funcs
// op slot Ids
// input size
// inputs (input count + input slots)
// params (param size(includes sizeof(param size)) + params)
//
// TODO: Adding an "expose" feature to the scripts would be cool. We could simply have a pair of "Type, param offset"
// to allow changing of a loaded scripts parameters
typedef struct _cranpa_script_t cranpa_script_t;

void cranpa_init(void);

// TODO: ops do a lot of memory management that takes away from their actual logic. Can we clean that up somehow?
unsigned long long cranpa_vm_buffer_size(unsigned long long memorySize, unsigned int maxActiveBuffers);
cranpa_vm_t* cranpa_vm_buffer_create(void* buffer, unsigned long long memorySize, unsigned int maxActiveBuffers);
void* cranpa_vm_alloc_chunk(cranpa_vm_t* vm, unsigned int slotId, unsigned long long memorySize);
void* cranpa_vm_get_chunk(cranpa_vm_t* vm, unsigned int slotId);

void cranpa_init_script(cranpa_script_t* script);
void cranpa_vm_execute_script(cranpa_vm_t* vm, cranpa_script_t* script);

#endif // __CRANBERRY_PROCEDURAL_ASSETS_H

#ifdef CRANPA_IMPLEMENTATION

#include "cranberry_math.h"

#include <stdint.h>
#include <string.h>

#ifdef CRANBERRY_DEBUG
#include <assert.h>
#include <stdio.h>

#define cranpa_assert(a) assert(a)
#define cranpa_log(a, ...) printf(a, __VA_ARGS__)
#else
#define cranpa_assert(a)
#define cranpa_log(a, ...)
#endif // CRANBERRY_DEBUG

#define cranpa_potentially_unused(a) (void)a
#define cranpa_alignment 16

typedef struct
{
	uint64_t allocChunkCount;
	uint64_t allocChunkSize;
	void* buffer;
} cranpa_allocator_t;

typedef struct
{
	uint16_t idx[3];
} cranpa_triangle_t;

/*
Buffer Layout:

- Memory -
cranpa_allocator
*/

void* cranpa_advance(void* ptr, uint64_t offset)
{
	return ((uint8_t*)ptr + offset);
}

uint64_t cranpa_allocator_buffer_size(uint64_t memorySize, uint32_t chunkCount)
{
	uint64_t chunkSize = memorySize / chunkCount;
	return sizeof(uint64_t) * 2 + chunkSize * chunkCount + cranpa_alignment * chunkCount;
}

void cranpa_allocator_buffer_create(void* buffer, uint64_t memorySize, uint32_t maxActiveBuffers)
{
	*(uint64_t*)buffer = maxActiveBuffers;
	*(uint64_t*)cranpa_advance(buffer, sizeof(uint64_t)) = memorySize / maxActiveBuffers;
}

cranpa_allocator_t cranpa_view_as_allocator(void* buffer)
{
	return (cranpa_allocator_t)
	{
		.allocChunkCount = *(uint64_t*)buffer,
		.allocChunkSize = *(uint64_t*)cranpa_advance(buffer, sizeof(uint64_t)),
		.buffer = cranpa_advance(buffer, sizeof(uint64_t) * 2)
	};
}

void* cranpa_allocator_get_chunk(cranpa_allocator_t* allocator, unsigned int slotId)
{
	// We don't have that many chunks!
	cranpa_assert(slotId < allocator->allocChunkCount);
	intptr_t chunkAddress = (intptr_t)cranpa_advance(allocator->buffer, slotId * allocator->allocChunkSize);
	chunkAddress += cranpa_alignment - chunkAddress % cranpa_alignment;
	return (void*)chunkAddress;
}

void cranpa_register_ops(void);
void cranpa_init(void)
{
	cranpa_register_ops();
}

// Memory
cranpa_allocator_t cranpa_vm_get_allocator(cranpa_vm_t* vm)
{
	return cranpa_view_as_allocator(vm);
}

unsigned long long cranpa_vm_buffer_size(unsigned long long memorySize, unsigned int maxActiveBuffers)
{
	cranpa_assert(memorySize % cranpa_alignment == 0); // Chunk size must be a multiple of our alignment!
	return cranpa_allocator_buffer_size(memorySize, maxActiveBuffers);
}

cranpa_vm_t* cranpa_vm_buffer_create(void* buffer, unsigned long long memorySize, unsigned int maxActiveBuffers)
{
	cranpa_allocator_buffer_create(buffer, memorySize, maxActiveBuffers);
	return (cranpa_vm_t*)buffer;
}

void* cranpa_vm_alloc_chunk(cranpa_vm_t* vm, unsigned int slotId, unsigned long long chunkSize)
{
	cranpa_potentially_unused(chunkSize);

	cranpa_allocator_t allocator = cranpa_vm_get_allocator(vm);
	cranpa_assert(chunkSize <= allocator.allocChunkSize);

	return cranpa_allocator_get_chunk(&allocator, slotId);
}

void* cranpa_vm_get_chunk(cranpa_vm_t* vm, unsigned int slotId)
{
	cranpa_allocator_t allocator = cranpa_vm_get_allocator(vm);
	return cranpa_allocator_get_chunk(&allocator, slotId);
}

// Interpreter
void cranpa_vm_execute_script(cranpa_vm_t* vm, cranpa_script_t* script)
{
	uint32_t opCount = *(uint32_t*)script;

	cranpa_opf_t* ops = (cranpa_opf_t*)cranpa_advance(script, sizeof(uint32_t));
	cranpa_opf_t* opEnd = ops + opCount;

	uint32_t* slotIds = (uint32_t*)cranpa_advance(script, sizeof(uint32_t) + opCount * sizeof(cranpa_opf_t));

	uint32_t* inputs = (uint32_t*)cranpa_advance(script, sizeof(uint32_t) + opCount * sizeof(cranpa_opf_t) + opCount * sizeof(uint32_t));
	uint32_t inputSize = *inputs;
	inputs++;

	uint32_t* params = (uint32_t*)cranpa_advance(script, sizeof(uint32_t) + opCount * sizeof(cranpa_opf_t) + opCount * sizeof(uint32_t) + inputSize);

	// Process first few ops
	for (uint32_t i = 0; i < opCount % 4; ++i)
	{
		uint32_t inputSlotCount = *inputs;
		ops[0](vm, slotIds[0], inputs + 1, inputSlotCount, params + 1);

		ops++;
		slotIds++;

		uint32_t paramSize = *params;
		params = (uint32_t*)cranpa_advance(params, paramSize);
		inputs = (uint32_t*)cranpa_advance(inputs, (inputSlotCount + 1) * sizeof(uint32_t));
	}

	while (ops != opEnd)
	{
		uint32_t inputSlotCount = *inputs;
		uint32_t paramSize = *params;

		ops[0](vm, slotIds[0], inputs + 1, inputSlotCount, params + 1);
		inputs = (uint32_t*)cranpa_advance(inputs, sizeof(uint32_t) * (inputSlotCount + 1));
		inputSlotCount = *inputs;
		params = cranpa_advance(params, paramSize);
		paramSize = *params;

		ops[1](vm, slotIds[1], inputs + 1, inputSlotCount, params + 1);
		inputs = (uint32_t*)cranpa_advance(inputs, sizeof(uint32_t) * (inputSlotCount + 1));
		inputSlotCount = *inputs;
		params = cranpa_advance(params, paramSize);
		paramSize = *params;

		ops[2](vm, slotIds[2], inputs + 1, inputSlotCount, params + 1);
		inputs = (uint32_t*)cranpa_advance(inputs, sizeof(uint32_t) * (inputSlotCount + 1));
		inputSlotCount = *inputs;
		params = cranpa_advance(params, paramSize);
		paramSize = *params;

		ops[3](vm, slotIds[3], inputs + 1, inputSlotCount, params + 1);
		inputs = (uint32_t*)cranpa_advance(inputs, sizeof(uint32_t) * (inputSlotCount + 1));
		inputSlotCount = *inputs;
		params = cranpa_advance(params, paramSize);

		slotIds += 4;
		ops += 4;
	}
}

// Ops
static cranpa_opf_t cranpa_op_table[cranpa_op_id_max];

void cranpa_init_script(cranpa_script_t* script)
{
	uint32_t opCount = *(uint32_t*)script;

	cranpa_opf_t* ops = (cranpa_opf_t*)cranpa_advance(script, sizeof(uint32_t));
	for (uint32_t i = 0; i < opCount; ++i)
	{
		intptr_t opId = (intptr_t)ops[i];
		cranpa_assert(opId < cranpa_op_id_max);
		ops[i] = cranpa_op_table[opId];
	}
}

void cranpa_op_circle(cranpa_vm_t* vm, unsigned int slotId, unsigned int* inputs, unsigned int inputCount, void* params)
{
	// Circle doesn't take any inputs
	cranpa_assert(inputCount == 0);

	cranpa_potentially_unused(inputs);
	cranpa_potentially_unused(inputCount);

	float segmentCount = *(float*)params;
	float radius = *((float*)params + 1);

	cranpa_assert(segmentCount >= 3.0f);

	uint32_t allocSize =
		sizeof(cranm_vec_t) // vertex count, triangle count, unused, unused
		+ ((uint32_t)segmentCount + 1) * sizeof(cranm_vec_t) // vertices
		+ (uint32_t)segmentCount * sizeof(cranpa_triangle_t); // triangles

	void* writeHead = cranpa_vm_alloc_chunk(vm, slotId, allocSize);
	
	// Write vert count and triangle count
	*(cranm_vec_t*)writeHead = (cranm_vec_t){.x = segmentCount + 1.0f, .y = segmentCount };
	writeHead = cranpa_advance(writeHead, sizeof(cranm_vec_t));

	{
		*(cranm_vec_t*)writeHead = (cranm_vec_t) { .x = 0.0f, .y = 0.0f, .z = 0.0f };
		writeHead = cranpa_advance(writeHead, sizeof(cranm_vec_t));

		// Mesh first
		float angleIncrement = 2.0f * cranm_pi / segmentCount;
		for (float segment = 0.0f; segment < segmentCount; segment += 1.0f)
		{
			*(cranm_vec_t*)writeHead = (cranm_vec_t) { .x = cosf(angleIncrement * segment) * radius, .y = sinf(angleIncrement * segment) * radius, .z = 0.0f };
			writeHead = cranpa_advance(writeHead, sizeof(cranm_vec_t));
		}
	}

	{
		// Triangles next
		uint32_t triangleCount = (uint32_t)segmentCount;
		for (uint16_t triangle = 0; triangle < triangleCount; triangle++)
		{
			*(cranpa_triangle_t*)writeHead = (cranpa_triangle_t){0, triangle + 1, triangle + 2};
			writeHead = cranpa_advance(writeHead, sizeof(cranpa_triangle_t));
		}
	}
}

void cranpa_op_translate(cranpa_vm_t* vm, unsigned int slotId, unsigned int* inputs, unsigned int inputCount, void* params)
{
	cranpa_assert(inputCount == 1);

	cranm_vec_t translation = *(cranm_vec_t*)params;

	void* readHead = cranpa_vm_get_chunk(vm, inputs[0]);
	cranm_vec_t counts = *(cranm_vec_t*)readHead;
	readHead = cranpa_advance(readHead, sizeof(cranm_vec_t));

	uint32_t vertCount = (uint32_t)counts.x;
	uint32_t triangleCount = (uint32_t)counts.y;
	uint32_t allocSize = sizeof(cranm_vec_t) + vertCount * sizeof(cranm_vec_t) + triangleCount * sizeof(cranpa_triangle_t);
	
	void* writeHead = cranpa_vm_alloc_chunk(vm, slotId, allocSize);
	// Write vert count and triangle count
	*(cranm_vec_t*)writeHead = counts;
	writeHead = cranpa_advance(writeHead, sizeof(cranm_vec_t));

	{
		// Mesh first
		for (uint32_t vert = 0; vert < vertCount; vert++)
		{
			*(cranm_vec_t*)writeHead = cranm_add3(*(cranm_vec_t*)readHead, translation);
			writeHead = cranpa_advance(writeHead, sizeof(cranm_vec_t));
			readHead = cranpa_advance(readHead, sizeof(cranm_vec_t));
		}
	}

	memcpy(writeHead, readHead, (uint32_t)triangleCount * sizeof(cranpa_triangle_t));
}

void cranpa_register_ops(void)
{
	cranpa_op_table[cranpa_op_id_circle] = cranpa_op_circle;
	cranpa_op_table[cranpa_op_id_translate] = cranpa_op_translate;
}

#endif // CRANPA_IMPLEMENTATION
