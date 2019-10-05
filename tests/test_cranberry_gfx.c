#include <stdio.h>

extern void test_backend(void);
extern void test_frontend(void);

int main(void)
{
	test_backend();
	test_frontend();
}
