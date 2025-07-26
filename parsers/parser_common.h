#ifndef _PARSER_COMMON_H
#define _PARSER_COMMON_H

#include <assert.h>
#include <errno.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include "../external/stb_ds.h"  // Dynamic arrays

//// Public interface -----------------------------------------------

// Prevent naming conflict with ../common.h
typedef struct {
  float x, y, z;
} _v3;

#define v3 _v3
typedef struct {
  const char *name;
  v3 Ka, Kd, Ks;
  char *map_Ka, *map_Kd, *map_Ks, *map_Ns;
  float Ns, Ni;
  float d;
  int illum;
} MtlParams;

// TODO: THIS IS OUTDATED. REPLACE THIS
typedef struct {
  v3 vs[3];
  v3 vts[3];
  MtlParams mtl;
} Face;
#undef v3

typedef struct {
  float x, y, z;
  float nx, ny, nz;
  float u, v;
} Vertex;

typedef struct {
  int nsides;
  int *idxs;    // A length `size` array of Vertex indices
} _Face;

typedef struct {
  Vertex *arr_vertices;
  _Face *arr_faces;
} Model;

#ifdef __CUDACC__
extern "C" void free_model(Model *model);
#else
void free_model(Model *model);
#endif


//// Private interface (only visible within parsers/) ---------------

#ifndef __CUDACC__

#ifdef NDEBUG
#define DEBUG(fmt, ...)
#else
#define DEBUG(fmt, ...) \
  printf(fmt "\n" __VA_OPT__(,) __VA_ARGS__)
#endif

#define ERROR(fmt, ...)	 \
  { printf("[rayt] ERROR at %s:%d: " fmt "\n", __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__); exit(EXIT_FAILURE); }

#define ERRORNO(fmt, ...) \
  { printf("[rayt] ERROR at %s:%d: %s %s: " fmt "\n", __FILE__, __LINE__, strerrorname_np(errno), strerror(errno) __VA_OPT__(,) __VA_ARGS__); exit(EXIT_FAILURE); }

typedef enum {false, true} bool;

char *read_whole_file(const char *filename);
void end_line(char **ptr);
void next_line(char **ptr);
bool consume(char **ptr, char *s);
bool consume_int(char **ptr, int *i);
void consume_any(char **ptr, char **s, int *len);
void swallow(char **ptr);

#endif

#endif
