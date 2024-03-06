PGCCFLAGS = -acc -Minfo=accel -Mneginfo
GCCFLAGS = -Wall -Wextra -Wpedantic -Wno-unused-but-set-variable -Wno-type-limits

all: clean gpu

debug: GCCFLAGS += -g -fanalyzer -fsanitize=address -fsanitize=undefined -fprofile-arcs -ftest-coverage
debug: cpu

cpu: render.c
	gcc $(GCCFLAGS) -o $@ $< -lm -DFOR_GPU=0

gpu: render.c
	pgcc $(PGCCFLAGS) -o $@ $< -lm -lcudart -DFOR_GPU=1

clean:
	rm -f gpu *.o