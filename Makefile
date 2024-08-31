# VARIABLES ---------------------------------------- 

BUILDDIR = build
OBJS = $(BUILDDIR)/stb_ds.o $(BUILDDIR)/stb_image.o $(BUILDDIR)/stb_image_write.o $(BUILDDIR)/load_obj.o
SRCS := $(wildcard *.c)
CFLAGS = -lm -Wall -Wextra -Wpedantic -Wno-unused-but-set-variable -Wno-unknown-pragmas -Wno-type-limits
CFLAGS_DEBUG = -g -pg -fanalyzer -fsanitize=address -fsanitize=undefined -ftest-coverage
GPU_FLAGS = -lcudart -acc -Minfo=accel -Mneginfo -O2

# DEFAULT TARGET ---------------------------------------- 

all: cpu

# PREBUILT TARGETS ---------------------------------------- 

# Fancy variable that uppercases a wildcard
UC = $(shell echo '$*' | tr '[:lower:]' '[:upper:]')

$(BUILDDIR)/stb_%.o: external/stb_%.h
	@echo "----- Building stb_$*.o -----"
	cd external; \
	cp stb_$*.h stb_$*.c; \
	gcc $(CFLAGS) -c stb_$*.c -DSTB_$(UC)_IMPLEMENTATION; \
	mv stb_$*.o ../$(BUILDDIR); \
	rm stb_$*.c;
	@echo -e "\n"

$(BUILDDIR)/load_obj.o: load_obj.c load_obj.h
	@echo "----- Building load_obj.o -----"	
	gcc $(CFLAGS) -c load_obj.c -o $@
	@echo -e "\n"

# USER-SPECIFIED TARGETS ---------------------------------------- 

cpu: CFLAGS += -O2

cpu: FORCE $(SRCS) $(OBJS)
	@echo "----- Building rayt, cpu version -----"	
	gcc $(CFLAGS) main.c $(OBJS) -o rayt-cpu -DFOR_GPU=0

cpu-debug: CFLAGS += $(CFLAGS_DEBUG)

cpu-debug: FORCE $(SRCS) $(OBJS)
	@echo "----- Building rayt, cpu debug version -----"	
	gcc $(CFLAGS) main.c $(OBJS) -o rayt-cpudbg -DFOR_GPU=0

gpu: FORCE $(SRCS) $(OBJS)
	@echo "----- Building rayt, gpu version -----"	
	pgcc $(GPU_FLAGS) main.c $(OBJS) -o rayt-gpu -DFOR_GPU=1

FORCE:
	mkdir -p build

clean:
	rm -f cpu cpu-debug gpu gmon.out *.gcno; \
	rm -rf build;
