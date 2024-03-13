#if FOR_GPU == 1
#include <cuda_runtime.h>
#endif

#include <assert.h>
#include <math.h>
#include <stdio.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "util.c"

typedef enum {
  MATTE, METAL, DIELECTRIC, LIGHT
} MaterialType;

typedef struct {
  MaterialType mtype;
  v3 color;
  /* For diffuse */
  float albedo; // Ratio of light that is diffusely reflected
  /* For metal */
  float reflectivity; // From 0 to 1, how reflective
  float fuzziness; // How fuzzy do the reflections look
  /* For dielectric */
  float eta; // Refractive index
} Material;

Material matte(v3 color, float albedo) {
  return (Material) {.mtype = MATTE, .color = color, .albedo = albedo};
}

Material metal(v3 color, float reflectivity, float fuzziness) {
  return (Material) {.mtype = METAL, .color = color, .reflectivity = reflectivity, .fuzziness = fuzziness};
}

Material dielectric(v3 color, float eta) {
  return (Material) {.mtype = DIELECTRIC, .color = color, .eta = eta};
}

Material light(v3 color) {
  return (Material) {.mtype = LIGHT, .color = color};
}

float dielectric_reflectance(float cos_theta, float eta_ratio) {
  float r0 = (1-eta_ratio) / (1+eta_ratio);
  r0 *= r0;
  return r0 + (1-r0) * powf(1-cos_theta, 5);
}

void interact_with_material(Material mat, v3 normal, v3 in_dir, v3 *out_dir, float *attenuation, unsigned int *X) {
  if (mat.mtype == MATTE) {
    *out_dir = add(normal, randsphere(X));
    *attenuation = mat.albedo;
  }
  else if (mat.mtype == METAL) {
    *out_dir = add(reflect(neg(in_dir), normal),
		   scl(randsphere(X), mat.fuzziness));
    *attenuation = 1.0 - mat.reflectivity;
  }
  else if (mat.mtype == DIELECTRIC) {
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
    *attenuation = 0;
  }
  else if (mat.mtype == LIGHT) {
    *attenuation = 1;
  }
}

v3 mix_with_color(Material mat, v3 color) {
  switch (mat.mtype) {
  case MATTE: return mul(mat.color, color);
  case METAL: return mul(mat.color, color);
  case DIELECTRIC: return mul(mat.color, color);
  case LIGHT: return mat.color;
  }
}
 
typedef enum {
  SPHERE
} PrimitiveType;

typedef struct {
  PrimitiveType ptype;
  Material mat;
  /* For the sphere */
  v3 center;
  float radius;
} Hittable;

//   _______  __        ______   .______        ___       __           _______.
//  /  _____||  |      /  __  \  |   _  \      /   \     |  |         /       |
// |  |  __  |  |     |  |  |  | |  |_)  |    /  ^  \    |  |        |   (----`
// |  | |_ | |  |     |  |  |  | |   _  <    /  /_\  \   |  |         \   \    
// |  |__| | |  `----.|  `--'  | |  |_)  |  /  _____  \  |  `----..----)   |   
//  \______| |_______| \______/  |______/  /__/     \__\ |_______||_______/

v3 sky_color;

#define MAX_HITTABLES 1024
int num_hts = 0;
Hittable hts[MAX_HITTABLES];

int width = 500;
int height = 300;

int max_recurse = 10;
int num_samples = 20; // How many times to compute per pixel
v3 lookfrom = (v3){0,0,0};
v3 lookat = (v3){0,5,0};
float vfov = 45.0; // An angle from 0 to 180 degrees
float defocus_angle = 0;
float defocus_radius;
float focal_len;
v3 vp_nw; // Position of top-left of viewport
v3 pix_du, pix_dv; // Viewport coordinate vectors, pixel-wise
v3 cam_u, cam_v; // Camera coordinate vectors

#pragma acc declare create(width,height)
#pragma acc declare create(sky_color)
#pragma acc declare create(hts[:MAX_HITTABLES],num_hts)
#pragma acc declare create(max_recurse)
#pragma acc declare create(num_samples)
#pragma acc declare create(lookfrom,lookat)
#pragma acc declare create(vfov)
#pragma acc declare create(defocus_angle,defocus_radius)
#pragma acc declare create(focal_len)
#pragma acc declare create(vp_nw)
#pragma acc declare create(pix_du,pix_dv)
#pragma acc declare create(cam_u,cam_v)

//  __    __  .___________. __   __      
// |  |  |  | |           ||  | |  |     
// |  |  |  | `---|  |----`|  | |  |     
// |  |  |  |     |  |     |  | |  |     
// |  `--'  |     |  |     |  | |  `----.
//  \______/      |__|     |__| |_______|

v3 gamma_correct(v3 col) {
  return (v3) {sqrtf(col.x), sqrtf(col.y), sqrtf(col.z)};
}

v3 ray_at(v3 origin, v3 dir, float t) {
  return add(origin, scl(dir, t));
}

void add_ht(Hittable ht) {
  assert(num_hts <= MAX_HITTABLES);
  hts[num_hts++] = ht;
}

void add_sphere(v3 center, float radius, Material mat) {
  add_ht((Hittable){.ptype = SPHERE, .mat = mat, .center = center, .radius = radius});
}

//      ___       ______ .___________. __    __       ___       __      
//     /   \     /      ||           ||  |  |  |     /   \     |  |     
//    /  ^  \   |  ,----'`---|  |----`|  |  |  |    /  ^  \    |  |     
//   /  /_\  \  |  |         |  |     |  |  |  |   /  /_\  \   |  |     
//  /  _____  \ |  `----.    |  |     |  `--'  |  /  _____  \  |  `----.
// /__/     \__\ \______|    |__|      \______/  /__/     \__\ |_______|

float intersect_sphere(v3 origin, v3 dir, v3 center, float radius) {
  // We want to solve the quadratic
  // <origin + t*dir - center, origin + t*dir - center> = radius^2
  // => <v+t*w, v+t*w> - radius^2
  //      = t^2<w,w> + 2t<v,w> + (<v,v>-radius^2)
  //      = 0,
  // where v = origin - center; w = dir.

  v3 v = sub(origin, center);
  v3 w = dir;

  // Coefficients of the quadratic ax^2+bx+c
  float a = dot(w,w);
  float b = 2.0 * dot(v,w);
  float c = dot(v,v) - radius*radius;

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

v3 get_normal(v3 point, Hittable ht) {
  switch (ht.ptype) {
  case SPHERE:
    return normalize(sub(point, ht.center));
  }
  return (v3){0,0,0};
}

void get_closest_intersection(v3 origin, v3 dir, float *t, v3 *v, Hittable *ht) {
  float t_closest = INFINITY;
  Hittable ht_closest;

  for (int i = 0; i < num_hts; i++) {
    Hittable ht = hts[i];
    float t;
    switch (ht.ptype) {
    case SPHERE:
      t = intersect_sphere(origin, dir, ht.center, ht.radius);
      if (t != NO_SOL && t < t_closest) {
    	ht_closest = ht;
    	t_closest = t;
      }
      break;
    }
  }

  if (t != NULL) *t = t_closest;
  if (v != NULL) *v = ray_at(origin, dir, t_closest);
  if (ht != NULL) *ht = ht_closest;
}

v3 get_color(v3 origin, v3 dir, int recursion_depth, unsigned int *X) {
  if (recursion_depth >= max_recurse)
    return BLACK;

  float t_closest;
  v3 intersection;
  Hittable ht_closest;
  get_closest_intersection(origin, dir, &t_closest, &intersection, &ht_closest);

  if (t_closest == INFINITY)
    return sky_color;

  if (ht_closest.mat.mtype == LIGHT)
    return ht_closest.mat.color;

  v3 normal = get_normal(intersection, ht_closest);

  v3 out_dir; float attenuation;
  interact_with_material(ht_closest.mat, normal, dir, &out_dir, &attenuation, X);

  // Offset intersection point slightly to prevent shadow acne
  intersection = ray_at(intersection, out_dir, 0.01);
  v3 col = scl(get_color(intersection, out_dir, recursion_depth+1, X), 1.0-attenuation);
  return mix_with_color(ht_closest.mat, col);
}

// The pixels are located at lattice points (x+0.5, y+0.5)
v3 get_pix_pos(int x, int y) {
  return add(vp_nw, add(scl(pix_du, x+0.5),
			scl(pix_dv, y+0.5)));
}

v3 compute_pixel_color(int i, int j, unsigned int *X) {
  v3 avg_col = {0,0,0};
  for (int k = 0; k < num_samples; k++) {
    v2 pix_offset = randsquare(X);
    v2 defocus_offset = scl2(randdisk(X), defocus_radius);
    v3 pix_pos = get_pix_pos(j+0.5+pix_offset.x, i+0.5+pix_offset.y);

    v3 defocused_lookfrom = add(lookfrom,
				add(scl(cam_u, defocus_offset.x),
				    scl(cam_v, defocus_offset.y)));
    v3 dir = sub(pix_pos, defocused_lookfrom);

    avg_col = add(avg_col, get_color(defocused_lookfrom, dir, 0, X));
  }
  avg_col = gamma_correct(scl(avg_col, 1.0/num_samples));
  return avg_col;
}

// .___  ___.      ___       __  .__   __. 
// |   \/   |     /   \     |  | |  \ |  | 
// |  \  /  |    /  ^  \    |  | |   \|  | 
// |  |\/|  |   /  /_\  \   |  | |  . `  | 
// |  |  |  |  /  _____  \  |  | |  |\   | 
// |__|  |__| /__/     \__\ |__| |__| \__| 

void render(unsigned char *pixels) {
  focal_len = dist(lookat, lookfrom);

  // Setup world and camera --------------------------------------------------
#define DEMO1
#ifdef DEMO1 // Spheres of different materials on the ground, adapted from Ray Tracing in a Weekend
  sky_color = (v3){0.5,0.7,1};
  //add_sphere((v3){0,-5000,0}, 5000, dielectric((v3){1,0.6,0.6}, 1.3));
  add_sphere((v3){0,-5000,0}, 5000, matte(GREEN, 0.2));
  add_sphere((v3){-2,1,3}, 1, dielectric(WHITE, 1.3));
  add_sphere((v3){0,1,3}, 1, matte(RED, 0.2));
  add_sphere((v3){2,1,3}, 1, metal((v3){0.6,0.6,1}, 1, 0.2));
  //add_sphere((v3){0,3,3}, 0.5, light((v3){5,5,5}));
  lookfrom = (v3){0,1.5,-3};
  lookat = (v3){0,1,3};
#endif
#ifdef DEMO2 // A whole bunch of spheres, adapted from Ray Tracing in a Weekend
  num_samples = 4;
  sky_color = (v3){0.5,0.7,1};
  add_matte_sphere((v3){0,-1000,0}, 1000, (v3){0.9,0.9,0.9}, 0.9);

  for (int i = -11; i < 11; i++) {
    for (int j = -11; j < 11; j++) {
      v3 center = {i+0.8*randunif(), 0.2, j+0.8*randunif()};
      if (dist(center, (v3){4,0.2,0}) <= 2)
      	continue;

      float choose_mat = randunif();
      float r = randunif();
      float g = randunif();
      float b = randunif();
      if (choose_mat < 0.8) {
	add_matte_sphere(center, 0.2, (v3){r,g,b}, 0.9);
      }
      else if (choose_mat < 0.95) {
	float fuzziness = randunif()/2;
	add_metal_sphere(center, 0.2, (v3){r,g,b}, 0, 0, 0.3, fuzziness);
      }
      else {
	float eta = randunif()+0.5;
	add_dielectric_sphere(center, 0.2, (v3){r,g,b}, eta);
      }
    }
  }

  add_dielectric_sphere((v3){0,1,0}, 1, (v3){1,1,1}, 1.5);
  add_metal_sphere((v3){-4,1,0}, 1, (v3){0.4,0.2,0.1}, 0, 0, 0.5, 0.2);
  add_matte_sphere((v3){4,1,0}, 1, (v3){0.7,0.6,0.5}, 0.9);

  vfov = 20;
  lookfrom = (v3){-13,2,-3};
  lookat = (v3){0,0,0};
  defocus_angle = 0;
  focal_len = 10;
#endif

  defocus_radius = 2 * focal_len * tanf(defocus_angle/360*2*PI / 2);

  v3 cam_vup = {0,1,0};
  // Camera basis vectors
  v3 cam_w = normalize(sub(lookat, lookfrom)); // Into the viewport
  cam_u = normalize(cross(cam_vup, cam_w)); // Right across the viewport
  cam_v = cross(cam_w, cam_u); // Up the viewport

  // Viewport
  float vp_h = 2 * tanf(vfov/360*2*PI / 2) * focal_len;
  float vp_w = vp_h * (float) width / height;
  v3 vp_u = scl(cam_u, vp_w);
  v3 vp_v = scl(cam_v, -vp_h);
  vp_nw = add(lookfrom,
		 add(scl(cam_w, focal_len),
		     add(scl(cam_u, -vp_w/2), scl(cam_v, vp_h/2))));
  pix_du = scl(vp_u, 1.0/width);
  pix_dv = scl(vp_v, 1.0/height);

  // Populate pixel data -------------------------------------------------- 
  // Note: the memory for pixels is allocated in main.c

#pragma acc update device(sky_color)
#pragma acc update device(width,height)
#pragma acc update device(hts[:num_hts],num_hts)
#pragma acc update device(max_recurse)
#pragma acc update device(num_samples)
#pragma acc update device(lookfrom,lookat)
#pragma acc update device(vfov)
#pragma acc update device(defocus_angle,defocus_radius)
#pragma acc update device(focal_len)
#pragma acc update device(vp_nw)
#pragma acc update device(pix_du,pix_dv)
#pragma acc update device(cam_u,cam_v)

  int rows_processed = 0;
#pragma acc kernels copy(pixels[:width*height*4]) copyin(rows_processed)
#pragma acc loop independent
  for (int i = 0; i < height; i++) {
#pragma acc loop independent
    for (int j = 0; j < width; j++) {
      unsigned int X = i*width + j;
      v3 col = colclamp(compute_pixel_color(i, j, &X));
      int idx = 4*(i*width + j);
      pixels[idx] = col.x * 255;
      pixels[idx+1] = col.y * 255;
      pixels[idx+2] = col.z * 255;
      pixels[idx+3] = 255;
    }
    rows_processed++;
#if FOR_GPU == 0
    printf("%d rows processed\n", rows_processed);
#endif
  }
}

int main() {
  // Increase stack size, apparently the raytracer uses it up really quickly...
  // Original stack size is 1024
  // ------------------------------------------------------------
#if FOR_GPU == 1
  cudaDeviceSetLimit(cudaLimitStackSize, 9192);
  size_t x;
  cudaDeviceGetLimit(&x, cudaLimitStackSize);
  printf("Stack size = %u\n", (unsigned)x);
#endif

  unsigned char *pixels = malloc(width*height*4);
  render(pixels);
  stbi_write_png("output.png", width, height, 4, pixels, width*sizeof(unsigned int));
  free(pixels);

  return 0;
}