# VARIABLES -----------------------------------------------------

BUILDDIR = build
OBJS = $(BUILDDIR)/stb_ds.o $(BUILDDIR)/stb_image.o $(BUILDDIR)/stb_image_write.o $(BUILDDIR)/load_obj.o
SRCS := $(wildcard *.c)
GCCFLAGS =
NVCCFLAGS =
# TODO DOESN'T WORK
ifdef DEBUG
  GCCFLAGS += -g -fanalyzer -fsanitize=address -fsanitize=undefined
  NVCCFLAGS += $(GCCFLAGS) -lineinfo
endif

# DEFAULT TARGET ------------------------------------------------

all: gpu

# PREBUILT TARGETS ----------------------------------------------

# Fancy variable that uppercases a wildcard
UC = $(shell echo '$*' | tr '[:lower:]' '[:upper:]')

$(BUILDDIR)/stb_%.o: external/stb_%.h
	@echo "----- Building stb_$*.o -----"
	cd external; \
	cp stb_$*.h stb_$*.c; \
	gcc -O3 -c stb_$*.c -DSTB_$(UC)_IMPLEMENTATION; \
	mv stb_$*.o ../$(BUILDDIR); \
	rm stb_$*.c;
	@echo -e "\n"

$(BUILDDIR)/load_obj.o: parsers/load_obj.c parsers/load_obj.h
	@echo "----- Building load_obj.o -----"
	gcc $(GCCFLAGS) -c parsers/load_obj.c -o $@
	@echo -e "\n"

# USER-SPECIFIED TARGETS ----------------------------------------

gpu: FORCE $(SRCS) $(OBJS)
	@echo "----- Building rayt -----"
	nvcc $(NVCCFLAGS) main.cu $(OBJS) -o rayt

FORCE:
	mkdir -p $(BUILDDIR)

clean:
	rm -f rayt \
	rm -rf $(BUILDDIR);
