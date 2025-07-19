#ifndef _AABB_C
#define _AABB_C

#include <math.h>
#include "common.h"

#define INTERVAL_EMPTY (Interval){INFINITY,INFINITY}

typedef struct {
  float low;
  float high;
} Interval;

__device__ bool is_empty_interval(Interval I) {
  return I.low == INFINITY && I.high == INFINITY;
}

__device__ Interval interval_within(float start, float speed, float low, float high) {
  if (speed == 0) {
    if (low < start && start < high)
      return (Interval){0, INFINITY};
    else
      return INTERVAL_EMPTY;
  }

  float t1 = (low-start) / speed;
  float t2 = (high-start) / speed;

  if (t1 > t2) {
    float tmp = t1;
    t1 = t2;
    t2 = tmp;
  }

  if (t1 == t2)
    return INTERVAL_EMPTY;

  return (Interval){t1,t2};
}

__device__ Interval interval_intersect(Interval I1, Interval I2) {
  if (is_empty_interval(I1) || is_empty_interval(I2))
    return INTERVAL_EMPTY;
  float M = fmaxf(I1.low, I2.low);
  float m = fminf(I1.high, I2.high);
  if (M >= m)
    return INTERVAL_EMPTY;
  return (Interval){M,m};
}

static inline aabb aabb_pad(aabb B) {
  float min_x = B.min_coords.x;
  float min_y = B.min_coords.y;
  float min_z = B.min_coords.z;
  float max_x = B.max_coords.x;
  float max_y = B.max_coords.y;
  float max_z = B.max_coords.z;
  return (aabb) {(v3){min_x-1, min_y-1, min_z-1},
		 (v3){max_x+1, max_y+1, max_z+1}};
}

static inline aabb aabb_union(aabb B1, aabb B2) {
  float min_x = fminf(B1.min_coords.x, B2.min_coords.x);
  float min_y = fminf(B1.min_coords.y, B2.min_coords.y);
  float min_z = fminf(B1.min_coords.z, B2.min_coords.z);
  float max_x = fmaxf(B1.max_coords.x, B2.max_coords.x);
  float max_y = fmaxf(B1.max_coords.y, B2.max_coords.y);
  float max_z = fmaxf(B1.max_coords.z, B2.max_coords.z);
  return (aabb) {(v3){min_x, min_y, min_z},
		 (v3){max_x, max_y, max_z}};
}

__device__ bool aabb_intersects(aabb aabb, v3 origin, v3 dir) {
  Interval I1 = interval_within(origin.x, dir.x, aabb.min_coords.x, aabb.max_coords.x);
  Interval I2 = interval_within(origin.y, dir.y, aabb.min_coords.y, aabb.max_coords.y);
  Interval I3 = interval_within(origin.z, dir.z, aabb.min_coords.z, aabb.max_coords.z);
  Interval intersection = interval_intersect(interval_intersect(I1,I2), I3);
  return !is_empty_interval(intersection);
}

void paabb(aabb aabb) {
  float min_x = aabb.min_coords.x;
  float max_x = aabb.max_coords.x;
  float min_y = aabb.min_coords.y;
  float max_y = aabb.max_coords.y;
  float min_z = aabb.min_coords.z;
  float max_z = aabb.max_coords.z;
  printf("<(%.2f,%.2f,%.2f), (%.2f,%.2f,%.2f)>", min_x, min_y, min_z, max_x, max_y, max_z);
}

#endif
