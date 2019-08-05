
#define CRANPR_IMPLEMENTATION
#define CRANPR_ENABLED
#include "../cranberry_profiler.h"

#include <stdlib.h>
#include <stdio.h>

void empty_test(void)
{
	cranpr_init();
	cranpr_write_to_file("profiling_test_empty.json");
	cranpr_terminate();
}

void full_test(void)
{
	cranpr_init();

	cranpr_begin("Full Profile", "Profiling...");
	for (int i = 0; i < 1024 * 100; i++)
	{
		cranpr_begin("Profiling!", "Profiling!");
		cranpr_end("Profiling!", "Profiling!");
		cranpr_event("Profiling!", "Event!");
	}
	cranpr_end("Full Profile", "Profiling...");

	cranpr_flush_thread_buffer();

	cranpr_begin("Profiling!", "Writing To File!");
	cranpr_write_to_file("profiling_test_full_1.json");
	cranpr_end("Profiling!", "Writing To File!");

	cranpr_flush_thread_buffer();
	cranpr_write_to_file("profiling_test_full_2.json");

	cranpr_terminate();
}

int main(void)
{
	empty_test();
	full_test();
	return 0;
}