#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


static inline bool is_prime(const uint64_t number)
{
	for (uint64_t i = 2; i <= sqrt(number); i++)
	{
		if (number % i == 0)
		{
			return false;
		}
	}

	return true;
}

int main(int argc, const char *argv[])
{
	uint64_t max = 0;
	if (argc < 2)
	{
		scanf("%zu", &max);
	}
	else
	{
		sscanf(argv[1], "%zu", &max);
	}

	if (max < 2 || argc > 1 && argv[1][0] == '-')
	{
		return EXIT_FAILURE;
	}

	for (uint64_t i = 2; i <= max; i++)
	{
		if (is_prime(i))
		{
			printf("%zu\n", i);
		}
	}

	return EXIT_SUCCESS;
}
