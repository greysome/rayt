#ifndef _LOAD_OBJ_H
#define _LOAD_OBJ_H

#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include "../external/stb_ds.h"

#ifndef __CUDACC__
typedef enum {false, true} bool;
#endif

// Prevent naming conflict with common.h
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

typedef struct {
  v3 vs[3], vts[3];
  MtlParams mtl;
} Face;

#ifdef __cplusplus
extern "C" Face *load_obj(const char *filename);
#else
Face *load_obj(const char *filename);
#endif

#undef v3

#endif
