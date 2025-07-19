//// This file contains the intersection of the set of functions
//// used in the different parts of the raytracer, in particular:
//// 1. General utility functions
//// 2. Vectors
//// 3. Random numbers
//// 4. Common types
//// 5. Color definitions

#ifndef _COMMON_H
#define _COMMON_H

//// GENERAL UTILITY FUNCTIONS ---------------------------------------

// Dynamic arrays
#include "external/stb_ds.h"

__host__ __device__ int clampi(int x, int low, int high) {
  if (x < low) return low;
  if (x > high) return high;
  return x;
}

cudaError_t _cudaerr;
#define CUDA_CATCH() if(_cudaerr=cudaGetLastError()){printf("[rayt] ERROR at %s:%d: CUDA: %s: %s\n",__FILE__,__LINE__,cudaGetErrorName(_cudaerr),cudaGetErrorString(_cudaerr));exit(1);}

//// VECTORS ---------------------------------------------------------
// For brevity, functions on v3s are given short names (e.g. add, scl).
// A suffix is added when operating on other types (e.g. scl2, clampcol).

typedef struct {
  float x;
  float y;
} v2;

// Used to store positions and colors
typedef struct {
  float x;
  float y;
  float z;
} v3;


// For debugging purposes
void pv3(v3 v) {
  printf("%f %f %f\n", v.x, v.y, v.z);
}

__host__ __device__ v3 add(v3 v, v3 w) {
  return (v3){v.x+w.x, v.y+w.y, v.z+w.z};
}

__host__ __device__ v3 neg(v3 v) {
  return (v3){-v.x,-v.y,-v.z};
}

__host__ __device__ v3 sub(v3 v, v3 w) {
  return (v3){v.x-w.x, v.y-w.y, v.z-w.z};
}

__host__ __device__ v3 mul(v3 v, v3 w) {
  return (v3){v.x*w.x, v.y*w.y, v.z*w.z};
}

// I wanted to name it div but apparently there's a name conflict
__host__ __device__ v3 divi(v3 v, v3 w) {
  return (v3){v.x/w.x, v.y/w.y, v.z/w.z};
}

__host__ __device__ v3 scl(v3 v, float c) {
  return (v3){v.x*c, v.y*c, v.z*c};
}

__host__ __device__ v2 scl2(v2 v, float c) {
  return (v2) {v.x*c, v.y*c};
}

__host__ __device__ float dot(v3 v, v3 w) {
  return v.x*w.x + v.y*w.y + v.z*w.z;
}

__host__ __device__ v3 cross(v3 v, v3 w) {
  return (v3){v.y*w.z - v.z*w.y,
              v.z*w.x - v.x*w.z,
              v.x*w.y - v.y*w.x};
}

__host__ __device__ float lensqr(v3 v) {
  return v.x*v.x + v.y*v.y + v.z*v.z;
}

__host__ __device__ float len(v3 v) {
  return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

__host__ __device__ v3 normalize(v3 v) {
  float c = len(v);
  return (v3){v.x/c, v.y/c, v.z/c};
}

__host__ __device__ float dist(v3 v, v3 w) {
  return len(sub(v,w));
}

__host__ __device__ v3 lerp(v3 v, v3 w, float t) {
  return add(scl(v,1-t), scl(w,t));
}

__host__ __device__ v3 refract(v3 dir, v3 normal, float eta_ratio) {
  dir = normalize(dir);
  v3 v = scl(dir, eta_ratio);
  float c = -dot(dir,normal);
  float x = eta_ratio*c - sqrt(1 - eta_ratio*eta_ratio*(1-c*c));
  v3 w = scl(normal, x);
  return add(v, w);
}

__host__ __device__ v3 reflect(v3 v, v3 w) {
  return normalize(sub(scl(w, 2*dot(v, w)), v));
}

__host__ __device__ v3 clampcol(v3 color) {
  if (color.x > 1) color.x = 1;
  if (color.y > 1) color.y = 1;
  if (color.z > 1) color.z = 1;
  if (color.x < 0) color.x = 0;
  if (color.y < 0) color.y = 0;
  if (color.z < 0) color.z = 0;
  return color;
}

__host__ __device__ v3 ray_at(v3 origin, v3 dir, float t) {
  return add(origin, scl(dir, t));
}

//// RANDOM NUMBERS --------------------------------------------------

__host__ __device__ float randunif(unsigned int *X) {
  // I use a straightforward linear congruential generator.
  // The choice of parameters m, a, c is taken from a table entry in the
  // Wikipedia page.
  *X = 134775813 * (*X) + 1;
  return ((float)*X) / 4294967296;
}

__host__ __device__ v2 randdisk(unsigned int *X) {
  while (true) {
    float x = randunif(X)*2-1;
    float y = randunif(X)*2-1;
    if (x*x + y*y <= 1)
      return (v2){x,y};
  }
}

__host__ __device__ v2 randsquare(unsigned int *X) {
  float x = randunif(X)*2-1;
  float y = randunif(X)*2-1;
  return (v2){x,y};
}

__host__ __device__ v3 randsphere(unsigned int *X) {
  float x = randunif(X)*2-1;
  float y = randunif(X)*2-1;
  float z = randunif(X)*2-1;
  return normalize((v3){x,y,z});
}

//// COMMON TYPES ----------------------------------------------------

typedef struct {
  int w, h;
  unsigned char *pixels;
} Image;

typedef enum {
  MATTE, METAL, DIELECTRIC, LIGHT
} MaterialType;

typedef struct {
  MaterialType type;

  //// For metals
  float reflectivity; // From 0 to 1, how reflective
  float fuzziness; // How fuzzy do the reflections look

  //// For dielectrics
  float eta; // Refractive index
} Material;

typedef enum {
  SOLID, CHECKER_ABS, CHECKER_REL, IMAGE
} TextureType;

typedef struct {
  TextureType type;

  //// For solids
  v3 color;

  //// For checkers
  float size;
  v3 color_even;
  v3 color_odd;

  //// For images
  int image_id;
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

  //// For spheres
  float r;

  //// For quads/triangles
  v3 u, v;
  v3 normal, w;
  float D;

  //// For SDFs
  int sdf_id;
  float size;
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

typedef struct {
  v3 sky_color;
  int w, h;

  int max_bounces;
  int samples_per_pixel;
  v3 lookfrom;
  v3 lookat;
  float vfov; // 0-180 degrees
  float defocus_angle;

  //// These variables are filled in by initialize_remaining_params():
  float focal_len;
  float defocus_radius;
  v3 viewport_topleft; // Viewport's northwest
  v3 pixel_du, pixel_dv; // Viewport coordinate vectors, pixel-wise
  v3 cam_u, cam_v; // Camera coordinate vectors
} RenderParams;

//// COLOR DEFINITIONS -----------------------------------------------

#define BLACK     ((v3){0,0,0})
#define WHITE     ((v3){1,1,1})
#define GRAY      ((v3){0.2,0.2,0.2})
#define LIGHTGRAY ((v3){0.5,0.5,0.5})
#define RED       ((v3){1,0,0})
#define GREEN     ((v3){0,1,0})
#define BLUE      ((v3){0,0,1})
#define YELLOW    ((v3){1,1,0})

#endif
