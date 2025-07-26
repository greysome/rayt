# VARIABLES -----------------------------------------------------

BUILDDIR = _build
OBJS = $(BUILDDIR)/stb_ds.o $(BUILDDIR)/stb_image.o $(BUILDDIR)/stb_image_write.o $(BUILDDIR)/load_obj.o $(BUILDDIR)/load_ply.o $(BUILDDIR)/parser_common.o
SRCS := $(wildcard *.c)
GCCFLAGS = -O3
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
	gcc $(GCCFLAGS) -c stb_$*.c -DSTB_$(UC)_IMPLEMENTATION; \
	mv stb_$*.o ../$(BUILDDIR); \
	rm stb_$*.c;
	@echo -e "\n"

$(BUILDDIR)/parser_common.o: parsers/parser_common.c parsers/parser_common.h
	@echo "----- Building parser_common.o -----"
	gcc $(GCCFLAGS) -c parsers/parser_common.c -o $@
	@echo -e "\n"

$(BUILDDIR)/load_%.o: parsers/load_%.c parsers/load_%.h
	@echo "----- Building load_$*.o -----"
	gcc $(GCCFLAGS) -c parsers/load_$*.c -o $@
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
