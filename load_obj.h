#ifndef _READ_OBJ_MTL_H
#define _READ_OBJ_MTL_H

#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdbool.h>
#include "external/stb_ds.h"

// I provide the bare definition of the vector type so that I don't
// have to include vector.c, which will lead to multiple definitions
// in the compilation of the main executable
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


Face *load_obj(const char *filename);

#undef v3

#endif