#include <math.h>
#include <stdio.h>

#define bool int
#define true 1
#define false 0
#define PI 3.141592653589793238462643383279502884197
#define NO_SOL -1

typedef struct {
  float x;
  float y;
} v2;

/* Used to store positions and colors */
typedef struct {
  float x;
  float y;
  float z;
} v3;

#define BLACK ((v3){0,0,0})
#define RED ((v3){1,0,0})
#define GREEN ((v3){0,1,0})
#define BLUE ((v3){0,0,1})
#define YELLOW ((v3){1,1,0})
#define WHITE ((v3){1,1,1})

// Vectors -------------------------------------------------------

void pv3(v3 v) {
  printf("%f %f %f\n", v.x, v.y, v.z);
}

v3 add(v3 v, v3 w) {
  return (v3){v.x+w.x, v.y+w.y, v.z+w.z};
}

v3 neg(v3 v) {
  return (v3){-v.x,-v.y,-v.z};
}

v3 sub(v3 v, v3 w) {
  return (v3){v.x-w.x, v.y-w.y, v.z-w.z};
}

v3 mul(v3 v, v3 w) {
  return (v3){v.x*w.x, v.y*w.y, v.z*w.z};
}

v3 scl(v3 v, float c) {
  return (v3){v.x*c, v.y*c, v.z*c};
}

v2 scl2(v2 v, float c) {
  return (v2) {v.x*c, v.y*c};
}

float dot(v3 v, v3 w) {
  return v.x*w.x + v.y*w.y + v.z*w.z;
}

v3 cross(v3 v, v3 w) {
  return (v3){v.y*w.z - v.z*w.y,
              v.z*w.x - v.x*w.z,
              v.x*w.y - v.y*w.x};
}

float lensqr(v3 v) {
  return v.x*v.x + v.y*v.y + v.z*v.z;
}

float len(v3 v) {
  return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

v3 normalize(v3 v) {
  float c = len(v); 
  return (v3){v.x/c, v.y/c, v.z/c};
}

float dist(v3 v, v3 w) {
  return len(sub(v,w));
}

v3 lerp(v3 v, v3 w, float t) {
  return add(scl(v,1-t), scl(w,t));
}

// r refers to the ratio between the two refractive indices
v3 refract(v3 dir, v3 normal, float r) {
  dir = normalize(dir);
  v3 v = scl(dir, r);
  float c = -dot(dir,normal);
  float x = r*c - sqrt(1 - r*r*(1-c*c));
  v3 w = scl(normal, x);
  return add(v, w);
}

v3 reflect(v3 v, v3 w) {
  return normalize(sub(scl(w, 2*dot(v, w)), v));
}

// Floats -------------------------------------------------------

float clampposf(float x) {
  if (x < 0) return 0;
  return x;
}

float lerpf(float start, float end, float t) {
  return (1-t)*start + t*end;
}

// Colors -------------------------------------------------------

int col_to_int(v3 col) {
  char r = col.x * 255;
  char g = col.y * 255;
  char b = col.z * 255;
  return r<<24 | g<<16 | b<<8 | 0xff;
}

v3 int_to_col(int col) {
  short r = col>>24 | 0xff;
  short g = col>>16 | 0xff;
  short b = col>>8 | 0xff;
  return (v3){r/255.0, g/255.0, b/255.0};
}

v3 colclamp(v3 color) {
  if (color.x > 1) color.x = 1;
  if (color.y > 1) color.y = 1;
  if (color.z > 1) color.z = 1;
  if (color.x < 0) color.x = 0;
  if (color.y < 0) color.y = 0;
  if (color.z < 0) color.z = 0;
  return color;
}

// Random -------------------------------------------------------

float randunif(unsigned int *X) {
  // I use a straightforward linear congruential generator
  // The choice of parameters m, a, c is taken from a table entry in the
  // Wikipedia page.
  *X = 134775813 * (*X) + 1;
  return ((float)*X) / 4294967296;
}

v2 randdisk(unsigned int *X) {
  while (true) {
    float x = randunif(X)*2-1;
    float y = randunif(X)*2-1;
    if (x*x + y*y <= 1)
      return (v2){x,y};
  }
}

v2 randsquare(unsigned int *X) {
  float x = randunif(X)*2-1;
  float y = randunif(X)*2-1;
  return (v2){x,y};
}

v3 randsphere(unsigned int *X) {
  float x = randunif(X)*2-1;
  float y = randunif(X)*2-1;
  float z = randunif(X)*2-1;
  return normalize((v3){x,y,z});
}