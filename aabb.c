#ifndef _AABB_C
#define _AABB_C

#include "vector.c"
#include "interval.c"

typedef struct {
  v3 min_coords;
  v3 max_coords;
} aabb;

static inline aabb aabb_union(aabb B1, aabb B2) {
  float min_x = fminf(B1.min_coords.x, B2.min_coords.x);
  float min_y = fminf(B1.min_coords.y, B2.min_coords.y);
  float min_z = fminf(B1.min_coords.z, B2.min_coords.z);
  float max_x = fmaxf(B1.max_coords.x, B2.max_coords.x);
  float max_y = fmaxf(B1.max_coords.y, B2.max_coords.y);
  float max_z = fmaxf(B1.max_coords.z, B2.max_coords.z);
  return (aabb) {.min_coords = (v3){min_x, min_y, min_z},
		 .max_coords = (v3){max_x, max_y, max_z}};
}

static inline bool aabb_intersects(aabb aabb, v3 origin, v3 dir) {
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