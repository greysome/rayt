#ifndef _RENDER_C
#define _RENDER_C

#if FOR_GPU == 1
#include <cuda_runtime.h>
#endif

#include <math.h>
#include <stdio.h>
#include "stb_image_write.h"
#include "vector.c"
#include "random.c"
#include "texture.c"
#include "material.c"
#include "primitive.c"
#include "lbvh.c"

// The maximum that max_recurse can be, lol
// The reason this is needed is because max_recurse isn't known at compile time,
// and VLAs as local variables aren't supported by pgcc...
#define MAX_MAX_RECURSE 64

v3 sky_color;

int stack_size = 1024;
int width = 1000;
int height = 700;

int max_recurse = 10;
int num_samples = 4; // How many times to compute per pixel
v3 lookfrom = {0,0,0};
v3 lookat = {0,5,0};
float vfov = 45.0; // An angle from 0 to 180 degrees
float defocus_angle = 0;
float defocus_radius;
float focal_len;
v3 vp_nw; // Position of top-left of viewport
v3 pix_du, pix_dv; // Viewport coordinate vectors, pixel-wise
v3 cam_u, cam_v; // Camera coordinate vectors

#pragma acc declare create(width,height)
#pragma acc declare create(sky_color)
#pragma acc declare create(max_recurse)
#pragma acc declare create(num_samples)
#pragma acc declare create(lookfrom,lookat)
#pragma acc declare create(vfov)
#pragma acc declare create(defocus_angle,defocus_radius)
#pragma acc declare create(focal_len)
#pragma acc declare create(vp_nw)
#pragma acc declare create(pix_du,pix_dv)
#pragma acc declare create(cam_u,cam_v)

v3 gamma_correct(v3 col) {
  return (v3) {sqrtf(col.x), sqrtf(col.y), sqrtf(col.z)};
}

v3 get_color(v3 origin, v3 dir, unsigned int *X) {
  v3 cur_origin = origin;
  v3 cur_dir = dir;
  v3 cur_color = WHITE;

  // Imitating a stack. I didn't use recursion because of the GPU's small stack.
  int depth = 0;
  Material materials[MAX_MAX_RECURSE];
  v3 tex_colors[MAX_MAX_RECURSE];

  for (; depth < max_recurse; depth++) {
    // Obtain closest intersection
    float t_closest;
    v3 intersection;
    Primitive obj_closest;
    get_closest_intersection(cur_origin, cur_dir, &t_closest, &intersection, &obj_closest);

    // If it is the sky, the rays stops bouncing and we proceed to mix together
    // all the color contributions
    if (t_closest == INFINITY) {
      cur_color = sky_color;
      depth--;
      goto fold_stack;
    }

    // Add material and texture color to the stack
    materials[depth] = obj_closest.mat;
    float u, v;
    get_uv(obj_closest, intersection, &u, &v);
    tex_colors[depth] = get_texture_color(obj_closest.tex, u, v, intersection);

    v3 normal = get_normal(obj_closest, intersection);
    v3 out_dir;
    interact_with_material(obj_closest.mat, normal, cur_dir, &out_dir, X);

    // Offset intersection point slightly to prevent shadow acne
    intersection = ray_at(intersection, out_dir, 0.01);
    cur_origin = intersection;
    cur_dir = out_dir;

    // If it is a light source, rays also stops bouncing
    // The light's material is already added onto the material stack, so we
    // don't have to set cur_color
    if (obj_closest.mat.type == LIGHT)
      goto fold_stack;
  }
  goto fold_stack;

fold_stack:
  for (; depth >= 0; depth--)
    cur_color = get_reflected_color(materials[depth], tex_colors[depth], cur_color);
  return cur_color;
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
    v3 pix_pos = get_pix_pos(j + 0.5 + pix_offset.x, i + 0.5 + pix_offset.y);

    v3 defocused_lookfrom = add(lookfrom,
				add(scl(cam_u, defocus_offset.x),
				    scl(cam_v, defocus_offset.y)));
    v3 dir = sub(pix_pos, defocused_lookfrom);

    avg_col = add(avg_col, get_color(defocused_lookfrom, dir, X));
  }
  avg_col = gamma_correct(scl(avg_col, 1.0/num_samples));
  return avg_col;
}

void initialize_constants() {
  focal_len = dist(lookat, lookfrom);
  defocus_radius = 2 * focal_len * tanf(defocus_angle/360*2*PI / 2);

  v3 cam_vup = {0,1,0};
  // Camera basis vectors
  v3 cam_w = normalize(sub(lookat, lookfrom)); // Into the viewport
  cam_u = normalize(cross(cam_vup, cam_w)); // Rigobj across the viewport
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
}

void render_to(unsigned char *pixels) {
#pragma acc update device(sky_color)
#pragma acc update device(width,height)
#pragma acc update device(max_recurse)
#pragma acc update device(num_samples)
#pragma acc update device(lookfrom,lookat)
#pragma acc update device(vfov)
#pragma acc update device(defocus_angle,defocus_radius)
#pragma acc update device(focal_len)
#pragma acc update device(vp_nw)
#pragma acc update device(pix_du,pix_dv)
#pragma acc update device(cam_u,cam_v)

#pragma acc update device(objs[:n_objs], nodes[:2*n_objs-1])

  int rows_processed = 0;
#pragma acc kernels copy(pixels[:width*height*4]) copyin(rows_processed)
#pragma acc loop independent
  for (int i = 0; i < height; i++) {
#pragma acc loop independent
    for (int j = 0; j < width; j++) {
      unsigned int X = i*width + j;
      v3 col = clampcol(compute_pixel_color(i, j, &X));
      int idx = 4*(i*width + j);
      pixels[idx] = col.x * 255;
      pixels[idx+1] = col.y * 255;
      pixels[idx+2] = col.z * 255;
      pixels[idx+3] = 255;
    }
    rows_processed++;
#if FOR_GPU == 0
    if (rows_processed % 10 == 0)
      printf("%d rows processed\n", rows_processed);
#endif
  }
}

void run() {
  // Increase GPU stack size if needed
#if FOR_GPU == 1
  cudaDeviceSetLimit(cudaLimitStackSize, stack_size);
  size_t x;
  cudaDeviceGetLimit(&x, cudaLimitStackSize);
  printf("Stack size = %u\n", (unsigned)x);
#endif

  // Some constants require computation
  initialize_constants();
  build_lbvh();

  /*
  int n = arrlen(objs);
  for (int i = 0; i < 2*n-1; i++) {
    lbvh_node node = nodes[i];
    if (node.is_leaf) {
      printf("Leaf(obj=%d, bbox=", node.idx_obj);
      paabb(node.aabb);
      printf("\n");
    }
    else {
      printf("Intermediate(split=%d, L=%d, R=%d, bbox=", node.idx_split, node.idx_left, node.idx_right);
      paabb(node.aabb);
      printf("\n");
    }
  }
  */

  assert(max_recurse <= MAX_MAX_RECURSE);

  unsigned char *pixels = malloc(width*height*4);
  render_to(pixels);
  stbi_write_png("output.png", width, height, 4, pixels, width*sizeof(unsigned int));
  free(pixels);

  free_textures();
}

#endif