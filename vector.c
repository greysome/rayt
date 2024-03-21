#ifndef _UTIL_C
#define _UTIL_C

#include <math.h>
#include <stdio.h>

#define PI 3.141592653589793238462643383279502884197

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

// Common color definitions
#define BLACK ((v3){0,0,0})
#define WHITE ((v3){1,1,1})
#define GRAY ((v3){0.2,0.2,0.2})
#define LIGHTGRAY ((v3){0.5,0.5,0.5})
#define RED ((v3){1,0,0})
#define GREEN ((v3){0,1,0})
#define BLUE ((v3){0,0,1})
#define YELLOW ((v3){1,1,0})

// VECTORS -------------------------------------------------------
// For brevity, functions on v3s are given short names (e.g. add, scl).
// A suffix is added when operating on different types (e.g. scl2, clampcol).

// For debugging purposes
void pv3(v3 v) {
  printf("%f %f %f\n", v.x, v.y, v.z);
}

static inline v3 add(v3 v, v3 w) {
  return (v3){v.x+w.x, v.y+w.y, v.z+w.z};
}

static inline v3 neg(v3 v) {
  return (v3){-v.x,-v.y,-v.z};
}

static inline v3 sub(v3 v, v3 w) {
  return (v3){v.x-w.x, v.y-w.y, v.z-w.z};
}

static inline v3 mul(v3 v, v3 w) {
  return (v3){v.x*w.x, v.y*w.y, v.z*w.z};
}

// I wanted to name it div but apparently there's a name conflict
static inline v3 divi(v3 v, v3 w) {
  return (v3){v.x/w.x, v.y/w.y, v.z/w.z};
}

static inline v3 scl(v3 v, float c) {
  return (v3){v.x*c, v.y*c, v.z*c};
}

static inline v2 scl2(v2 v, float c) {
  return (v2) {v.x*c, v.y*c};
}

static inline float dot(v3 v, v3 w) {
  return v.x*w.x + v.y*w.y + v.z*w.z;
}

static inline v3 cross(v3 v, v3 w) {
  return (v3){v.y*w.z - v.z*w.y,
              v.z*w.x - v.x*w.z,
              v.x*w.y - v.y*w.x};
}

static inline float lensqr(v3 v) {
  return v.x*v.x + v.y*v.y + v.z*v.z;
}

static inline float len(v3 v) {
  return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

static inline v3 normalize(v3 v) {
  float c = len(v); 
  return (v3){v.x/c, v.y/c, v.z/c};
}

static inline float dist(v3 v, v3 w) {
  return len(sub(v,w));
}

static inline v3 lerp(v3 v, v3 w, float t) {
  return add(scl(v,1-t), scl(w,t));
}

static inline v3 refract(v3 dir, v3 normal, float eta_ratio) {
  dir = normalize(dir);
  v3 v = scl(dir, eta_ratio);
  float c = -dot(dir,normal);
  float x = eta_ratio*c - sqrt(1 - eta_ratio*eta_ratio*(1-c*c));
  v3 w = scl(normal, x);
  return add(v, w);
}

static inline v3 reflect(v3 v, v3 w) {
  return normalize(sub(scl(w, 2*dot(v, w)), v));
}

static inline v3 clampcol(v3 color) {
  if (color.x > 1) color.x = 1;
  if (color.y > 1) color.y = 1;
  if (color.z > 1) color.z = 1;
  if (color.x < 0) color.x = 0;
  if (color.y < 0) color.y = 0;
  if (color.z < 0) color.z = 0;
  return color;
}

v3 ray_at(v3 origin, v3 dir, float t) {
  return add(origin, scl(dir, t));
}

#endif