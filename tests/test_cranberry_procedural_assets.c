#include <stdint.h>
#include <stdlib.h>

#include <stdio.h>

#include <assert.h>

#define CRANBERRY_DEBUG

#define CRANM_SSE
#include "../cranberry_math.h"

#define CRANPR_ENABLED
#define CRANPR_IMPLEMENTATION
#include "../cranberry_profiler.h"

#define CRANPA_IMPLEMENTATION
#include "../cranberry_procedural_assets.h"

const char* cranpa_test_script_ops[] = 
{
	[cranpa_op_id_circle] = "circle",
	[cranpa_op_id_translate] = "translate"
};

const char* cranpa_test_script_basic =
"circle 0 [] [10000.0,10.0]\n"
"translate 1 [0] [10.0,10.0,10.0,0.0f]\n";

cranpa_script_t* cranpa_test_parse_script(const char* script)
{
	typedef enum
	{
		cranpa_parse_op,
		cranpa_parse_id,
		cranpa_parse_inputs,
		cranpa_parse_params
	} cranpa_test_parser_state_e;

	intptr_t* opChunk = (intptr_t*)malloc(sizeof(cranpa_opf_t) * 100);
	uint32_t opCount = 0;

	uint32_t* opIdChunk = (uint32_t*)malloc(sizeof(uint32_t) * 100);
	uint32_t opIdCount = 0;

	uint32_t* inputChunk = (uint32_t*)malloc(sizeof(uint32_t) * (100 * 2 + 1));
	uint32_t inputWriteCount = 0;

	uint8_t* paramChunk = (uint8_t*)malloc(sizeof(uint32_t) * 100 * 2);
	uint32_t paramWriteSize = 0;

	cranpa_test_parser_state_e parserState = cranpa_parse_op;

	char const* prevIter = script;
	for (const char* iter = script; *iter != '\0'; iter++)
	{
		if (*iter == ' ' || *iter == '\n')
		{
			switch (parserState)
			{
			case cranpa_parse_op:
			{
				cranpa_op_id_e op = cranpa_op_id_max;
				for (uint32_t i = 0; i < cranpa_op_id_max; i++)
				{
					const char* opIter = cranpa_test_script_ops[i];
					const char* parsedOpIter = prevIter;
					for (; parsedOpIter != iter && *opIter != '\0'; parsedOpIter++, opIter++)
					{
						if (*opIter != *parsedOpIter)
						{
							break;
						}
					}

					// We found our op
					if (*opIter == '\0' && parsedOpIter == iter)
					{
						op = i;
						break;
					}
				}
				cranpa_assert(op != cranpa_op_id_max);

				opChunk[opCount++] = (intptr_t)op;

				parserState = cranpa_parse_id;
				prevIter = iter + 1; // We want to skip the ' '
			}
			break;

			case cranpa_parse_id:
			{
				uint32_t id = (uint32_t)atoi(prevIter);
				opIdChunk[opIdCount++] = id;

				parserState = cranpa_parse_inputs;
				prevIter = iter + 1; // We want to skip the ' '
			}
			break;

			case cranpa_parse_inputs:
			{
				assert(*prevIter == '[');

				uint32_t inputCount = 0;

				const char* lastNumberStart = prevIter + 1;
				const char* numberIter = lastNumberStart;
				while (1)
				{
					if (*numberIter == ',' || (*numberIter == ']' && *lastNumberStart != ']'))
					{
						inputChunk[inputWriteCount + inputCount + 1] = (uint32_t)atoi(lastNumberStart);
						inputCount++;

						lastNumberStart = numberIter + 1;
					}

					if (*numberIter == ']')
					{
						break;
					}

					numberIter++;
				}

				inputChunk[inputWriteCount] = inputCount;
				inputWriteCount += inputCount + 1;

				parserState = cranpa_parse_params;
				prevIter = iter + 1; // We want to skip " "
			}
			break;

			case cranpa_parse_params:
			{
				assert(*prevIter == '[');

				uint32_t paramSize = 0;

				const char* lastNumberStart = prevIter + 1;
				const char* numberIter = lastNumberStart;
				while (1)
				{
					if (*numberIter == ',' || *numberIter == ']')
					{
						*(float*)(paramChunk + (paramWriteSize + paramSize + sizeof(uint32_t))) = (float)atof(lastNumberStart);
						paramSize += sizeof(float);

						lastNumberStart = numberIter + 1;
					}

					if (*numberIter == ']')
					{
						break;
					}

					numberIter++;
				}

				*(uint32_t*)(paramChunk + paramWriteSize) = paramSize + sizeof(uint32_t);
				paramWriteSize += paramSize + sizeof(uint32_t);

				parserState = cranpa_parse_op;
				prevIter = iter + 1; // We want to skip " "
			}
			break;
			}
		}
	}

	void* scriptBuffer = 
		malloc(sizeof(uint32_t) 
			+ sizeof(cranpa_opf_t) * opCount 
			+ sizeof(uint32_t) * opCount 
			+ sizeof(uint32_t)
			+ sizeof(uint32_t) * inputWriteCount 
			+ paramWriteSize);

	uint8_t* compiledScript = scriptBuffer;
	*(uint32_t*)compiledScript = opCount;
	compiledScript += sizeof(uint32_t);
	memcpy(compiledScript, opChunk, sizeof(cranpa_opf_t) * opCount);
	compiledScript += sizeof(cranpa_opf_t) * opCount;
	memcpy(compiledScript, opIdChunk, sizeof(uint32_t) * opCount);
	compiledScript += sizeof(uint32_t) * opCount;

	*(uint32_t*)compiledScript = inputWriteCount * sizeof(uint32_t) + sizeof(uint32_t);
	compiledScript += sizeof(uint32_t);
	memcpy(compiledScript, inputChunk, sizeof(uint32_t) * inputWriteCount);
	compiledScript += sizeof(uint32_t) * inputWriteCount;
	memcpy(compiledScript, paramChunk, paramWriteSize);


	free(opChunk);
	free(opIdChunk);
	free(inputChunk);
	free(paramChunk);

	return scriptBuffer;
}

void cranpa_test(void)
{
	// Plain old vm construction
	{
		cranpr_begin("cranpa_test", "vm construct");

		unsigned long long bufferSize = cranpa_vm_buffer_size(1 << 16, 10);
		void* buffer = malloc((size_t)bufferSize);
		cranpa_vm_t* vm = cranpa_vm_buffer_create(buffer, 1 << 16, 10);
		free(buffer);

		cranpr_end("cranpa_test", "vm construct");
	}

	// Alloc a chunk and write to it
	{
		cranpr_begin("cranpa_test", "vm chunk");

		unsigned long long bufferSize = cranpa_vm_buffer_size(1 << 16, 10);
		void* buffer = malloc((size_t)bufferSize);
		cranpa_vm_t* vm = cranpa_vm_buffer_create(buffer, 1 << 16, 10);

		void* chunk = cranpa_vm_alloc_chunk(vm, 0, 100);
		memset(chunk, 0, 1 << 16 / 10);
		free(buffer);

		cranpr_end("cranpa_test", "vm chunk");
	}

	// Parse the test_basic script and run it
	{
		cranpr_begin("cranpa_test", "vm basic script");

		unsigned long long bufferSize = cranpa_vm_buffer_size(1 << 20, 4);
		void* buffer = malloc((size_t)bufferSize);
		cranpa_vm_t* vm = cranpa_vm_buffer_create(buffer, 1 << 20, 4);

		cranpa_script_t* script = cranpa_test_parse_script(cranpa_test_script_basic);
		cranpa_init_script(script);

		cranpr_begin("cranpa_test", "execute");
		cranpa_vm_execute_script(vm, script);
		cranpr_end("cranpa_test", "execute");

		free(script);
		free(buffer);

		cranpr_end("cranpa_test", "vm basic script");
	}
}

int main()
{
	cranpr_init();
	cranpa_init();

	cranpr_begin("main", "cranp_test");
	cranpa_test();
	cranpr_end("main", "cranp_test");

	cranpr_flush_thread_buffer();
	cranpr_write_to_file("test_procedural_assets.json");
	cranpr_terminate();

	return 0;
}
