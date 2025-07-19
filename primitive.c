#ifndef _PRIMITIVE_C
#define _PRIMITIVE_C

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include "common.h"
#include "aabb.c"
#include "material.c"

#define NO_SOL -INFINITY
#define PI 3.141592653589793238462643383279502884197

typedef struct {
  float t;
  v3 p;
  Primitive prim;
  float u, v;
} HitRecord;

void add_sphere(RenderScene *scene, v3 pos, float r, Material mat, Texture tex) {
  Primitive prim;
  prim.type = SPHERE;
  prim.mat = mat;
  prim.tex = tex;
  prim.pos = pos;
  prim.r = r;
  arrpush(scene->prims, prim);
}

void add_quad(RenderScene *scene, v3 pos, v3 u, v3 v, Material mat, Texture tex) {
  Primitive prim;
  prim.type = QUAD;
  prim.mat = mat;
  prim.tex = tex;
  prim.pos = pos;
  prim.u = u;
  prim.v = v;
  v3 n = cross(u,v); v3 normal = normalize(n);
  prim.normal = normalize(n);
  prim.w = scl(n, 1.0/lensqr(n));
  prim.D = dot(normal, pos);
  arrpush(scene->prims, prim);
}

void add_triangle(RenderScene *scene, v3 p1, v3 p2, v3 p3, Material mat, Texture tex) {
  v3 u = sub(p2,p1);
  v3 v = sub(p3,p1);
  v3 n = cross(u,v);
  v3 normal = normalize(n);
  Primitive prim;
  prim.type = TRIANGLE;
  prim.mat = mat;
  prim.tex = tex;
  prim.pos = p1;
  prim.u = u;
  prim.v = v;
  prim.normal = normal;
  prim.w = scl(n, 1.0/lensqr(n));
  prim.D = dot(normal, p1);
  arrpush(scene->prims, prim);
}

void add_sdf(RenderScene *scene, v3 p, float size, Material mat, Texture tex) {
  // SDFs don't support non-solid textures, because how do you compute uv coordinates!?
  assert(tex.type == SOLID);
  Primitive prim = (Primitive){.type = SDF, .mat = mat, .tex = tex, .pos = p,
			      .size = size};
  prim.type = SDF;
  prim.mat = mat;
  prim.tex = tex;
  prim.pos = p;
  prim.size = size;
  arrpush(scene->prims, prim);
}

__device__ v3 get_sdf_coords(Primitive prim, v3 p) {
  return scl(sub(p, prim.pos), 1.0/prim.size);
}
// Hard-coded example, change this later
__device__ float sdf1(v3 p) {
  return sqrtf(0.3*p.x*p.x + p.y*p.y + p.z*p.z) - 1;
}
aabb sdf1_aabb = {(v3){-2,-1,-1}, (v3){2,1,1}};

aabb get_aabb(Primitive prim) {
  if (prim.type == SPHERE) {
    float x = prim.pos.x;
    float y = prim.pos.y;
    float z = prim.pos.z;
    float r = prim.r;
    v3 min_coords = {x-r, y-r, z-r};
    v3 max_coords = {x+r, y+r, z+r};
    return (aabb){min_coords, max_coords};
  }

  else if (prim.type == QUAD || prim.type == TRIANGLE) {
    v3 v = prim.pos;
    v3 w = add(prim.pos, add(prim.u, prim.v));
    float min_x = fminf(v.x, w.x);
    float min_y = fminf(v.y, w.y);
    float min_z = fminf(v.z, w.z);
    float max_x = fmaxf(v.x, w.x);
    float max_y = fmaxf(v.y, w.y);
    float max_z = fmaxf(v.z, w.z);
    return aabb_pad((aabb){(v3){min_x,min_y,min_z},
			   (v3){max_x,max_y,max_z}});
  }

  else if (prim.type == SDF) {
    return (aabb) {add(scl(sdf1_aabb.min_coords, prim.size), prim.pos),
		   add(scl(sdf1_aabb.max_coords, prim.size), prim.pos)};
  }

  return (aabb){(v3){-INFINITY,-INFINITY,-INFINITY},
		(v3){INFINITY,INFINITY,INFINITY}};
}

__device__ v3 get_normal(Primitive prim, v3 p) {
  if (prim.type == SPHERE)
    return normalize(sub(p, prim.pos));

  else if (prim.type == QUAD || prim.type == TRIANGLE)
    return prim.normal;

  else if (prim.type == SDF) {
    float e = 0.0001;
    float dx = sdf1(get_sdf_coords(prim, add(p, (v3){e,0,0}))) - sdf1(get_sdf_coords(prim, add(p, (v3){-e,0,0})));
    float dy = sdf1(get_sdf_coords(prim, add(p, (v3){0,e,0}))) - sdf1(get_sdf_coords(prim, add(p, (v3){0,-e,0})));
    float dz = sdf1(get_sdf_coords(prim, add(p, (v3){0,0,e}))) - sdf1(get_sdf_coords(prim, add(p, (v3){0,0,-e})));
    return normalize((v3){dx,dy,dz});
  }

  return (v3){0,0,0};
}

__device__ void get_uv(Primitive prim, v3 p, float *u, float *v) {
  if (prim.type == SPHERE) {
    // Get the unit normal vector pointing from sphere center to p
    p = normalize(sub(p, prim.pos));
    // Compute cartesian -> spherical coordinates, taking r=1
    *v = acosf(p.y) / PI;
    *u = atan2f(p.z, p.x)/(2*PI) + 0.5;
  }

  else if (prim.type == QUAD || prim.type == TRIANGLE) {
    *u = dot(prim.w, cross(p, prim.v));
    *v = dot(prim.w, cross(prim.u, p));
  }
}

__device__ float get_intersection(Primitive prim, v3 origin, v3 dir, float *u, float *v) {
  if (prim.type == SPHERE) {
    // We want to solve the quadratic
    // <origin + t*dir - center, origin + t*dir - center> = radius^2
    // => <V+t*W, V+t*W> - radius^2
    //      = t^2<W,W> + 2t<V,W> + (<V,V>-radius^2)
    //      = 0,
    // where V = origin - center; W = dir.

    v3 V = sub(origin, prim.pos);
    v3 W = dir;

    // Coefficients of the quadratic ax^2+bx+c
    float a = dot(W,W);
    float b = 2.0 * dot(V,W);
    float c = dot(V,V) - prim.r*prim.r;

    float discr = b*b - 4*a*c;
    if (discr >= 0) {
      // Return the smallest positive solution, or NO_SOL
      float sqrt_discr = sqrtf(discr);
      float t1 = (-b-sqrt_discr) / (2.0*a);
      float t2 = (-b+sqrt_discr) / (2.0*a);
      if (t1 > 0) {
        v3 p = ray_at(origin, dir, t1);
        get_uv(prim, p, u, v);
        return t1;
      }
      else if (t2 > 0) {
        v3 p = ray_at(origin, dir, t2);
        get_uv(prim, p, u, v);
        return t2;
      }
      else return NO_SOL;
    }
    else
      return NO_SOL;
  }

  else if (prim.type == QUAD || prim.type == TRIANGLE) {
    // First we find where the ray intersects the plane containing the quad
    // If n is the normal vector out of the quad, we solve for t the following equation:
    // <n, origin+t*dir> = <n, prim.pos>
    // => t*<n,dir> = D - <n, origin>,
    // where D = <n, prim.pos> and the normal n are stored in the primect's data
    float denom = dot(prim.normal, dir);
    // Ray is parallel to plane
    if (fabs(denom) < 0.000001)
      return NO_SOL;
    float t = (prim.D - dot(prim.normal, origin)) / denom;

    // Find coefficients of intersection points wrt u and v
    v3 p = sub(ray_at(origin, dir, t), prim.pos);
    get_uv(prim, p, u, v);

    // The quad corresponds to the region of the plane where
    // 0 <= u, v <= 1.
    if (*u < 0 || *u > 1 || *v < 0 || *v > 1)
      return NO_SOL;
    // The triangle corresponds to the region of the plane where
    // 0 <= u, v <= 1 and u+v <= 1
    if (prim.type == TRIANGLE && *u+*v > 1)
      return NO_SOL;

    return t;
  }

  else if (prim.type == SDF) {
    float l = len(dir);
    dir = normalize(dir);
    float cur_t = 0;
    v3 cur_p = origin;

    while (true) {
      float sd = sdf1(get_sdf_coords(prim, cur_p));

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
