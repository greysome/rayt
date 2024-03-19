#ifndef _HITTABLE_C
#define _HITTABLE_C

#include <assert.h>
#define STB_DS_IMPLEMENTATION
#include "vector.c"
#include "random.c"
#include "aabb.c"
#include "material.c"

#define NO_SOL -1
 
typedef enum {
  SPHERE
} PrimitiveType;

typedef struct {
  PrimitiveType type;
  Material mat;
  v3 pos;
  unsigned int morton_code; // For LBVH purposes
  union {
    float r;
  };
} Primitive;


#define MAX_OBJS 100000

Primitive objs[MAX_OBJS];
int n_objs = 0;
#pragma acc declare create(objs[:MAX_OBJS])

void add_sphere(v3 pos, float r, Material mat) {
  Primitive obj = (Primitive){.type = SPHERE, .mat = mat, .pos = pos, .r = r};
  objs[n_objs++] = obj;
}



//                            ,d      ,d                                    
//                            88      88                                    
//    ,adPPYb,d8  ,adPPYba, MM88MMM MM88MMM ,adPPYba, 8b,dPPYba, ,adPPYba,  
//   a8"    `Y88 a8P_____88   88      88   a8P_____88 88P'   "Y8 I8[    ""  
//   8b       88 8PP"""""""   88      88   8PP""""""" 88          `"Y8ba,   
//   "8a,   ,d88 "8b,   ,aa   88,     88,  "8b,   ,aa 88         aa    ]8I  
//    `"YbbdP"Y8  `"Ybbd8"'   "Y888   "Y888 `"Ybbd8"' 88         `"YbbdP"'  
//    aa,    ,88                                                            
//     "Y8bbdP"                                                             


float get_intersection(Primitive obj, v3 origin, v3 dir) {
  // -------------------------------------------------- 
  // Sphere

  if (obj.type == SPHERE) {
    // We want to solve the quadratic
    // <origin + t*dir - center, origin + t*dir - center> = radius^2
    // => <v+t*w, v+t*w> - radius^2
    //      = t^2<w,w> + 2t<v,w> + (<v,v>-radius^2)
    //      = 0,
    // where v = origin - center; w = dir.

    v3 v = sub(origin, obj.pos);
    v3 w = dir;

    // Coefficients of the quadratic ax^2+bx+c
    float a = dot(w,w);
    float b = 2.0 * dot(v,w);
    float c = dot(v,v) - obj.r*obj.r;

    float discr = b*b - 4*a*c;
    if (discr >= 0) {
      // Return the smallest positive solution, or NO_SOL
      float sqrt_discr = sqrtf(discr);
      float t1 = (-b-sqrt_discr) / (2.0*a);
      float t2 = (-b+sqrt_discr) / (2.0*a);
      if (t1 > 0) return t1;
      else if (t2 > 0) return t2;
      else return NO_SOL;
    }
    else
      return NO_SOL;
  }
}

aabb get_aabb(Primitive obj) {
  // -------------------------------------------------- 
  // Sphere

  if (obj.type == SPHERE) {
    float x = obj.pos.x;
    float y = obj.pos.y;
    float z = obj.pos.z;
    float r = obj.r;
    v3 min_coords = {x-r, y-r, z-r};
    v3 max_coords = {x+r, y+r, z+r};
    return (aabb){min_coords, max_coords};
  }
}

v3 get_normal(v3 point, Primitive obj) {
  // -------------------------------------------------- 
  // Sphere

  if (obj.type == SPHERE)
    return normalize(sub(point, obj.pos));
}

/*
void get_closest_intersection(v3 origin, v3 dir, float *t, v3 *v, Primitive *obj) {
  float t_closest = INFINITY;
  Primitive obj_closest;

  for (int i = 0; i < arrlen(objs); i++) {
    Primitive obj = objs[i];
    float t;
    switch (obj.type) {
    case SPHERE:
      t = get_intersection(obj, origin, dir);
      if (t != NO_SOL && t < t_closest) {
    	obj_closest = obj;
    	t_closest = t;
      }
      break;
    }
  }

  if (t != NULL) *t = t_closest;
  if (v != NULL) *v = ray_at(origin, dir, t_closest);
  if (obj != NULL) *obj = obj_closest;
}
*/

#endif