#ifndef _HITTABLE_C
#define _HITTABLE_C

#if FOR_GPU == 1
#include <cuda_runtime.h>
#endif

#include <math.h>
#include <stdio.h>
#include <assert.h>
#define STB_DS_IMPLEMENTATION
#include "vector.c"
#include "random.c"
#include "aabb.c"
#include "material.c"

#define NO_SOL -INFINITY
 
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
  float t;
  v3 p;
  Primitive obj;
  float u, v;
} HitRecord;


#define MAX_OBJS 100000

Primitive objs[MAX_OBJS];
int n_objs = 0;
#pragma acc declare create(objs[:MAX_OBJS])

void add_sphere(v3 pos, float r, Material mat, Texture tex) {
  Primitive obj = (Primitive){.type = SPHERE, .mat = mat, .tex = tex, .pos = pos, .r = r};
  objs[n_objs++] = obj;
}

void add_quad(v3 pos, v3 u, v3 v, Material mat, Texture tex) {
  v3 n = cross(u,v);
  v3 normal = normalize(n);
  Primitive obj = (Primitive){.type = QUAD, .mat = mat, .tex = tex, .pos = pos,
			      .u = u, .v = v,
			      .normal = normal, .w = scl(n, 1.0/lensqr(n)),
			      .D = dot(normal, pos)};
  objs[n_objs++] = obj;
}

void add_triangle(v3 p1, v3 p2, v3 p3, Material mat, Texture tex) {
  v3 u = sub(p2,p1);
  v3 v = sub(p3,p1);
  v3 n = cross(u,v);
  v3 normal = normalize(n);
  Primitive obj = (Primitive){.type = TRIANGLE, .mat = mat, .tex = tex, .pos = p1,
			      .u = u, .v = v,
			      .normal = normal, .w = scl(n, 1.0/lensqr(n)),
			      .D = dot(normal, p1)};
  objs[n_objs++] = obj;
}

void add_sdf(v3 p, float size, Material mat, Texture tex) {
  // SDFs don't support non-solid textures, because how do you compute uv coordinates!?
  assert(tex.type == SOLID);
  Primitive obj = (Primitive){.type = SDF, .mat = mat, .tex = tex, .pos = p,
			      .size = size};
  objs[n_objs++] = obj;
}

void free_textures() {
  for (int i = 0; i < n_objs; i++) {
    Primitive obj = objs[i];
    if (obj.tex.type == IMAGE) {
#if FOR_GPU == 0
      free(obj.tex.pixels);
#else
      cudaError_t err;
      if (err = cudaFree(obj.tex.pixels))
	printf("TEXTURE: failed to free image data -- %s\n", cudaGetErrorString(err));
#endif
    }
  }
}

v3 get_sdf_coords(Primitive obj, v3 p) {
  return scl(sub(p, obj.pos), 1.0/obj.size);
}

float sdf1(v3 p);
aabb sdf1_aabb;


//                            ,d      ,d                                    
//                            88      88                                    
//    ,adPPYb,d8  ,adPPYba, MM88MMM MM88MMM ,adPPYba, 8b,dPPYba, ,adPPYba,  
//   a8"    `Y88 a8P_____88   88      88   a8P_____88 88P'   "Y8 I8[    ""  
//   8b       88 8PP"""""""   88      88   8PP""""""" 88          `"Y8ba,   
//   "8a,   ,d88 "8b,   ,aa   88,     88,  "8b,   ,aa 88         aa    ]8I  
//    `"YbbdP"Y8  `"Ybbd8"'   "Y888   "Y888 `"Ybbd8"' 88         `"YbbdP"'  
//    aa,    ,88                                                            
//     "Y8bbdP"                                                             

aabb get_aabb(Primitive obj) {
  if (obj.type == SPHERE) {
    float x = obj.pos.x;
    float y = obj.pos.y;
    float z = obj.pos.z;
    float r = obj.r;
    v3 min_coords = {x-r, y-r, z-r};
    v3 max_coords = {x+r, y+r, z+r};
    return (aabb){min_coords, max_coords};
  }

  else if (obj.type == QUAD || obj.type == TRIANGLE) {
    v3 v = obj.pos;
    v3 w = add(obj.pos, add(obj.u, obj.v));
    float min_x = fminf(v.x, w.x);
    float min_y = fminf(v.y, w.y);
    float min_z = fminf(v.z, w.z);
    float max_x = fmaxf(v.x, w.x);
    float max_y = fmaxf(v.y, w.y);
    float max_z = fmaxf(v.z, w.z);
    return aabb_pad((aabb){(v3){min_x,min_y,min_z},
			   (v3){max_x,max_y,max_z}});
  }

  else if (obj.type == SDF) {
    return (aabb) {add(scl(sdf1_aabb.min_coords, obj.size), obj.pos),
		   add(scl(sdf1_aabb.max_coords, obj.size), obj.pos)};
  }

  return (aabb){(v3){-INFINITY,-INFINITY,-INFINITY},
		(v3){INFINITY,INFINITY,INFINITY}};
}

v3 get_normal(Primitive obj, v3 p) {
  if (obj.type == SPHERE)
    return normalize(sub(p, obj.pos));

  else if (obj.type == QUAD || obj.type == TRIANGLE)
    return obj.normal;

  else if (obj.type == SDF) {
    float e = 0.0001;
    float dx = sdf1(get_sdf_coords(obj, add(p, (v3){e,0,0}))) - sdf1(get_sdf_coords(obj, add(p, (v3){-e,0,0})));
    float dy = sdf1(get_sdf_coords(obj, add(p, (v3){0,e,0}))) - sdf1(get_sdf_coords(obj, add(p, (v3){0,-e,0})));
    float dz = sdf1(get_sdf_coords(obj, add(p, (v3){0,0,e}))) - sdf1(get_sdf_coords(obj, add(p, (v3){0,0,-e})));
    return normalize((v3){dx,dy,dz});
  }

  return (v3){0,0,0};
}

void get_uv(Primitive obj, v3 p, float *u, float *v) {
  if (obj.type == SPHERE) {
    // Get the unit normal vector pointing from sphere center to p
    p = normalize(sub(p, obj.pos));
    // Compute cartesian -> spherical coordinates, taking r=1
    *v = acosf(p.y) / PI;
    *u = atan2f(p.z, p.x)/(2*PI) + 0.5;
  }

  else if (obj.type == QUAD || obj.type == TRIANGLE) {
    *u = dot(obj.w, cross(p, obj.v));
    *v = dot(obj.w, cross(obj.u, p));
  }
}

float get_intersection(Primitive obj, v3 origin, v3 dir, float *u, float *v) {
  if (obj.type == SPHERE) {
    // We want to solve the quadratic
    // <origin + t*dir - center, origin + t*dir - center> = radius^2
    // => <V+t*W, V+t*W> - radius^2
    //      = t^2<W,W> + 2t<V,W> + (<V,V>-radius^2)
    //      = 0,
    // where V = origin - center; W = dir.

    v3 V = sub(origin, obj.pos);
    v3 W = dir;

    // Coefficients of the quadratic ax^2+bx+c
    float a = dot(W,W);
    float b = 2.0 * dot(V,W);
    float c = dot(V,V) - obj.r*obj.r;

    float discr = b*b - 4*a*c;
    if (discr >= 0) {
      // Return the smallest positive solution, or NO_SOL
      float sqrt_discr = sqrtf(discr);
      float t1 = (-b-sqrt_discr) / (2.0*a);
      float t2 = (-b+sqrt_discr) / (2.0*a);
      if (t1 > 0) {
	v3 p = ray_at(origin, dir, t1);
	get_uv(obj, p, u, v);
	return t1;
      }
      else if (t2 > 0) {
	v3 p = ray_at(origin, dir, t2);
	get_uv(obj, p, u, v);
	return t2;
      }
      else return NO_SOL;
    }
    else
      return NO_SOL;
  }

  else if (obj.type == QUAD || obj.type == TRIANGLE) {
    // First we find where the ray intersects the plane containing the quad
    // If n is the normal vector out of the quad, we solve for t the following equation:
    // <n, origin+t*dir> = <n, obj.pos>
    // => t*<n,dir> = D - <n, origin>,
    // where D = <n, obj.pos> and the normal n are stored in the object's data
    float denom = dot(obj.normal, dir);
    // Ray is parallel to plane
    if (fabs(denom) < 0.000001)
      return NO_SOL;
    float t = (obj.D - dot(obj.normal, origin)) / denom;

    // Find coefficients of intersection points wrt u and v
    v3 p = sub(ray_at(origin, dir, t), obj.pos);
    get_uv(obj, p, u, v);

    // The quad corresponds to the region of the plane where
    // 0 <= u, v <= 1.
    if (*u < 0 || *u > 1 || *v < 0 || *v > 1)
      return NO_SOL;
    // The triangle corresponds to the region of the plane where
    // 0 <= u, v <= 1 and u+v <= 1
    if (obj.type == TRIANGLE && *u+*v > 1)
      return NO_SOL;

    return t;
  }

  else if (obj.type == SDF) {
    float l = len(dir);
    dir = normalize(dir);
    float cur_t = 0;
    v3 cur_p = origin;

    while (true) {
      float sd = sdf1(get_sdf_coords(obj, cur_p));

      if (fabs(sd) < 0.00001) return cur_t / l;
      if (sd > 100000) return NO_SOL;

      // Make sure the ray always marches forward
      cur_t += fabs(sd);
      cur_p = ray_at(origin, dir, cur_t);
    }
  }

  return NO_SOL;
}

#endif