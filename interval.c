#ifndef _INTERVAL_C
#define _INTERVAL_C

#include <math.h>
#include <stdbool.h>

#define INTERVAL_EMPTY (Interval){INFINITY,INFINITY}

typedef struct {
  float low;
  float high;
} Interval;

static inline bool is_empty_interval(Interval I) {
  return I.low == INFINITY && I.high == INFINITY;
}

static inline Interval interval_within(float start, float speed, float low, float high) {
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

static inline Interval interval_intersect(Interval I1, Interval I2) {
  if (is_empty_interval(I1) || is_empty_interval(I2))
    return INTERVAL_EMPTY;
  float M = fmaxf(I1.low, I2.low);
  float m = fminf(I1.high, I2.high);
  if (M >= m)
    return INTERVAL_EMPTY;
  return (Interval){M,m};
}

#endif