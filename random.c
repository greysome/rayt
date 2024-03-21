#ifndef _RANDOM_C
#define _RANDOM_C

#include <stdbool.h>
#include "vector.c"

static inline float randunif(unsigned int *X) {
  // I use a straightforward linear congruential generator.
  // The choice of parameters m, a, c is taken from a table entry in the
  // Wikipedia page.
  *X = 134775813 * (*X) + 1;
  return ((float)*X) / 4294967296;
}

static inline v2 randdisk(unsigned int *X) {
  while (true) {
    float x = randunif(X)*2-1;
    float y = randunif(X)*2-1;
    if (x*x + y*y <= 1)
      return (v2){x,y};
  }
}

static inline v2 randsquare(unsigned int *X) {
  float x = randunif(X)*2-1;
  float y = randunif(X)*2-1;
  return (v2){x,y};
}

static inline v3 randsphere(unsigned int *X) {
  float x = randunif(X)*2-1;
  float y = randunif(X)*2-1;
  float z = randunif(X)*2-1;
  return normalize((v3){x,y,z});
}

#endif