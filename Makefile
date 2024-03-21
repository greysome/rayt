# VARIABLES ---------------------------------------- 

OBJS = stb_image.o stb_image_write.o
SRCS = vector.c random.c interval.c aabb.c texture.c material.c primitive.c lbvh_node.c lbvh.c render.c main.c
CFLAGS = -Wall -Wextra -Wpedantic -Wno-unused-but-set-variable -Wno-unknown-pragmas -Wno-type-limits
CFLAGS_PLUS_DEBUG = $(CFLAGS) -g -pg -fanalyzer -fsanitize=address -fsanitize=undefined -ftest-coverage
GPU_FLAGS = -acc -Minfo=accel -Mneginfo -O2

# TARGETS ---------------------------------------- 

all: cpu

stb_image.o: stb_image.h
	cp stb_image.h stb_image.c
	gcc $(CFLAGS) -c stb_image.c -DSTB_IMAGE_IMPLEMENTATION
	rm stb_image.c

stb_image_write.o: stb_image_write.h
	cp stb_image_write.h stb_image_write.c
	gcc $(CFLAGS) -c stb_image_write.c -DSTB_IMAGE_WRITE_IMPLEMENTATION
	rm stb_image_write.c

cpu: CFLAGS += -O2

cpu: $(SRCS) $(OBJS)
	gcc $(CFLAGS) main.c $(OBJS) -o cpu -lm -DFOR_GPU=0

cpu-debug: $(SRCS) $(OBJS)
	gcc $(CFLAGS_PLUS_DEBUG) main.c $(OBJS) -o cpu-debug -lm -DFOR_GPU=0

gpu: $(SRCS) $(OBJS)
	pgcc $(GPU_FLAGS) main.c $(OBJS) -o gpu -lcudart -lm -DFOR_GPU=1

clean:
	rm -f *.o cpu cpu-debug gpu gmon.out
