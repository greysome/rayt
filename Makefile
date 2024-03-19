# VARIABLES ---------------------------------------- 

SRCS = vector.c random.c interval.c aabb.c material.c primitive.c lbvh_node.c lbvh.c render.c main.c
CFLAGS = -Wall -Wextra -Wpedantic -Wno-unused-but-set-variable -Wno-unknown-pragmas -Wno-type-limits -O
CFLAGS_PLUS_DEBUG = $(CFLAGS) -g -pg -fanalyzer -fsanitize=address -fsanitize=undefined -ftest-coverage
GPU_FLAGS = -acc -Minfo=accel -Mneginfo -O2

# TARGETS ---------------------------------------- 

all: cpu

cpu: $(SRCS)
	gcc $(CFLAGS) main.c -o cpu -lm -DFOR_GPU=0

cpu-debug: $(SRCS)
	gcc $(CFLAGS_PLUS_DEBUG) main.c -o cpu-debug -lm -DFOR_GPU=0

gpu: $(SRCS)
	pgcc $(GPU_FLAGS) main.c -o gpu -lcudart -lm -DFOR_GPU=1

clean:
	rm -f *.gcda *.gcno cpu cpu-debug gpu gmon.out
