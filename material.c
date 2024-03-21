#ifndef _MATERIAL_C
#define _MATERIAL_C

#include <stdbool.h>
#include "vector.c"
#include "random.c"
#include "texture.c"

typedef enum {
  MATTE, METAL, DIELECTRIC, LIGHT
} MaterialType;

typedef struct {
  MaterialType type;
  union {
    /* For lambertians */
    float albedo; // Ratio of light that is absorbed

    /* For metal */
    struct {
      float reflectivity; // From 0 to 1, how reflective
      float fuzziness; // How fuzzy do the reflections look
    };

    /* For dielectrics */
    float eta; // Refractive index
  };
} Material;

Material matte(float albedo) {
  return (Material) {.type = MATTE, .albedo = albedo};
}

Material metal(float reflectivity, float fuzziness) {
  return (Material) {.type = METAL, .reflectivity = reflectivity, .fuzziness = fuzziness};
}

Material dielectric(float eta) {
  return (Material) {.type = DIELECTRIC, .eta = eta};
}

Material light() {
  return (Material) {.type = LIGHT};
}

static inline float dielectric_reflectance(float cos_theta, float eta_ratio) {
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

static inline v3 get_reflected_color(Material mat, v3 tex_color, v3 in_color) {
  switch (mat.type) {
  case MATTE: return scl(mul(tex_color, in_color), 1.0-mat.albedo);
  case METAL: return scl(mul(tex_color, in_color), mat.reflectivity);
  case DIELECTRIC: return mul(tex_color, in_color);
  case LIGHT: return tex_color;
  }
  return BLACK;
}

#endif