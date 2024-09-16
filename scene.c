#ifndef _SCENE_C
#define _SCENE_C

#include <stdbool.h>
#include "external/stb_ds.h"
#include "vector.c"




typedef struct {
  int w, h;
  unsigned char *pixels;
} Image;





typedef enum {
  MATTE, METAL, DIELECTRIC, LIGHT
} MaterialType;


typedef struct {
  MaterialType type;
  union {
    /* For metal */
    struct {
      float reflectivity; // From 0 to 1, how reflective
      float fuzziness; // How fuzzy do the reflections look
    };

    /* For dielectrics */
    float eta; // Refractive index
  };
} Material;


typedef enum {
  SOLID, CHECKER_ABS, CHECKER_REL, IMAGE
} TextureType;


typedef struct {
  TextureType type;
  union {
    /* For solids */
    v3 color;

    /* For checker */
    struct {
      float size;
      v3 color_even;
      v3 color_odd;
    };

    /* For image */
    int image_id;
  };
} Texture;
 

typedef enum {
  SPHERE, QUAD, TRIANGLE, SDF
} PrimitiveType;


typedef struct {
  PrimitiveType type;
  Material mat;
  Texture tex;
  v3 pos;
  unsigned int morton_code; // For LBVH purposes
  union {
    /* Sphere */
    float r;

    /* Quad/triangle */
    struct {
      v3 u, v;
      v3 normal, w;
      float D;
    };

    /* SDF */
    struct {
      int sdf_id;
      float size;
    };
  };
} Primitive;




typedef struct {
  v3 min_coords;
  v3 max_coords;
} aabb;

typedef struct {
  aabb aabb;
  bool is_leaf;
  int idx_prim; // Index into sorted primitive list (by Morton code)
  int idx_split; // Index at which to split primitives into two subnodes
  int idx_left; // Index of left subnode in node list
  int idx_right; // Index of right subnode in node list
} lbvh_node;




typedef struct {
  Image *images;
  Primitive *prims;
  lbvh_node *nodes;
} RenderScene;

#endif
