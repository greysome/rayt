#ifndef _MATERIAL_C
#define _MATERIAL_C

#include <stdbool.h>
#include "vector.c"
#include "random.c"

typedef enum {
  MATTE, METAL, DIELECTRIC, LIGHT
} MaterialType;

typedef struct {
  MaterialType type;
  v3 color;
  union {
    /* For lambertians */
    float albedo; // Ratio of light that is absorbed
    /* For metal */
    struct {
      float reflectivity; // From 0 to 1, how reflective
      float fuzziness; // How fuzzy do the reflections look
    };
    /* For dielectric */
    float eta; // Refractive index
  };
} Material;

Material matte(v3 color, float albedo) {
  return (Material) {.type = MATTE, .color = color, .albedo = albedo};
}

Material metal(v3 color, float reflectivity, float fuzziness) {
  return (Material) {.type = METAL, .color = color, .reflectivity = reflectivity, .fuzziness = fuzziness};
}

Material dielectric(v3 color, float eta) {
  return (Material) {.type = DIELECTRIC, .color = color, .eta = eta};
}

Material light(v3 color) {
  return (Material) {.type = LIGHT, .color = color};
}

inline float dielectric_reflectance(float cos_theta, float eta_ratio) {
  float r0 = (1-eta_ratio) / (1+eta_ratio);
  r0 *= r0;
  return r0 + (1-r0) * powf(1-cos_theta, 5);
}

void interact_with_material(Material mat, v3 normal, v3 in_dir, v3 *out_dir, unsigned int *X) {
  if (mat.type == MATTE) {
    *out_dir = add(normal, randsphere(X));
  }
  else if (mat.type == METAL) {
    *out_dir = add(reflect(neg(in_dir), normal),
		   scl(randsphere(X), mat.fuzziness));
  }
  else if (mat.type == DIELECTRIC) {
    bool inside = dot(in_dir, normal) > 0;
    float eta_ratio = inside ? mat.eta : 1.0/mat.eta;
    if (inside)
      normal = neg(normal);

    float cos_angle = fabsf(dot(in_dir, normal) / len(in_dir));
    float sin_angle = sqrtf(1 - cos_angle*cos_angle);
    float reflectance = dielectric_reflectance(cos_angle, mat.eta);
    // If Snell's equation is not solvable, then reflect (i.e. total internal reflection).
    // Else refract with a probability determined by reflectance at the angle.
    if (eta_ratio * sin_angle > 1 || reflectance > randunif(X))
      *out_dir = reflect(neg(in_dir), normal);
    else
      *out_dir = refract(in_dir, normal, eta_ratio);
  }
}

inline v3 mix_with_color(Material mat, v3 color) {
  switch (mat.type) {
  case MATTE: return scl(mul(mat.color, color), 1.0-mat.albedo);
  case METAL: return scl(mul(mat.color, color), mat.reflectivity);
  case DIELECTRIC: return mul(mat.color, color);
  case LIGHT: return mat.color;
  }
}

#endif