#ifndef _RENDER_C
#define _RENDER_C

#if FOR_GPU == 1
#include <cuda_runtime.h>
#endif

#include <math.h>
#include <stdio.h>
#include "external/stb_image_write.h"
#include "vector.c"
#include "random.c"
#include "texture.c"
#include "material.c"
#include "primitive.c"
#include "lbvh.c"

// The maximum that RenderParams.max_bounces can be, lol
// The reason is because I originally wanted a VLA with size
// RenderParams.max_bounces in get_color(), but this is not suppoted
// by pgcc...
#define MAX_MAX_BOUNCES 64

typedef struct {
  v3 sky_color; 
  int w, h;

  int max_bounces;
  int samples_per_pixel;
  v3 lookfrom;
  v3 lookat;
  float vfov; // 0-180 degrees
  float defocus_angle;
  float defocus_radius;
  float focal_len;
  v3 viewport_topleft; // Viewport's northwest
  v3 pixel_du, pixel_dv; // Viewport coordinate vectors, pixel-wise
  v3 cam_u, cam_v; // Camera coordinate vectors
} RenderParams;



v3 gamma_correct(v3 col) {
  return (v3) {sqrtf(col.x), sqrtf(col.y), sqrtf(col.z)};
}



v3 get_color(v3 origin, v3 dir, unsigned int *X, RenderParams *params) {
  v3 cur_origin = origin;
  v3 cur_dir = dir;
  v3 cur_color = WHITE;

  // Imitating a stack. I didn't use recursion because of the GPU's small stack.
  int depth = 0;
  Material materials[MAX_MAX_BOUNCES];
  v3 tex_colors[MAX_MAX_BOUNCES];

  for (; depth < params->max_bounces; depth++) {
    // Obtain closest intersection
    HitRecord hr;
    if (n_objs > 0) get_closest_intersection(cur_origin, cur_dir, &hr);
    else hr.t = INFINITY;

    // If it is the sky, the rays stops bouncing and we proceed to mix together
    // all the color contributions
    if (hr.t == INFINITY) {
      cur_color = params->sky_color;
      depth--;
      goto fold_stack;
    }

    // Add material and texture color to the stack
    materials[depth] = hr.obj.mat;
    tex_colors[depth] = get_texture_color(hr.obj.tex, hr.u, hr.v, hr.p);

    v3 normal = get_normal(hr.obj, hr.p);
    v3 out_dir;
    interact_with_material(hr.obj.mat, normal, cur_dir, &out_dir, X);

    // Offset intersection point slightly to prevent shadow acne
    hr.p = ray_at(hr.p, out_dir, 0.01);
    cur_origin = hr.p;
    cur_dir = out_dir;

    // If it is a light source, rays also stops bouncing
    // The light's material is already added onto the material stack, so we
    // don't have to set cur_color
    if (hr.obj.mat.type == LIGHT)
      goto fold_stack;
  }
  goto fold_stack;

fold_stack:
  for (; depth >= 0; depth--)
    cur_color = get_reflected_color(materials[depth], tex_colors[depth], cur_color);
  return cur_color;
}



// The pixels are located at lattice points (x+0.5, y+0.5)
v3 get_pix_pos(int x, int y, RenderParams *params) {
  return add(params->viewport_topleft,
	     add(scl(params->pixel_du, x+0.5),
		 scl(params->pixel_dv, y+0.5)));
}



v3 compute_pixel_color(int i, int j, unsigned int *X, RenderParams *params) {
  v3 avg_col = {0,0,0};
  for (int k = 0; k < params->samples_per_pixel; k++) {
    v2 pix_offset = randsquare(X);
    v2 defocus_offset = scl2(randdisk(X), params->defocus_radius);
    v3 pix_pos = get_pix_pos(j + 0.5 + pix_offset.x,
			     i + 0.5 + pix_offset.y,
			     params);

    v3 defocused_lookfrom = add(params->lookfrom,
				add(scl(params->cam_u, defocus_offset.x),
				    scl(params->cam_v, defocus_offset.y)));
    v3 dir = sub(pix_pos, defocused_lookfrom);

    avg_col = add(avg_col, get_color(defocused_lookfrom, dir, X, params));
  }
  avg_col = gamma_correct(scl(avg_col, 1.0/params->samples_per_pixel));
  return avg_col;
}



void initialize_constants(RenderParams *params) {
  params->focal_len = dist(params->lookat, params->lookfrom);
  params->defocus_radius = 2 * params->focal_len * tanf(params->defocus_angle/360*2*PI
							/ 2);

  v3 cam_vup = {0,1,0};
  // Camera basis vectors
  v3 cam_w = normalize(sub(params->lookat, params->lookfrom)); // Into the viewport
  params->cam_u = normalize(cross(cam_vup, cam_w)); // Right across the viewport
  params->cam_v = cross(cam_w, params->cam_u); // Up the viewport

  // Viewport
  float vp_h = 2 * tanf(params->vfov/360*2*PI / 2) * params->focal_len;
  float vp_w = vp_h * (float) params->w / params->h;
  v3 vp_u = scl(params->cam_u, vp_w);
  v3 vp_v = scl(params->cam_v, -vp_h);
  params->viewport_topleft = add(params->lookfrom,
			 add(scl(cam_w, params->focal_len),
			     add(scl(params->cam_u, -vp_w/2),
				 scl(params->cam_v, vp_h/2))));
  params->pixel_du = scl(vp_u, 1.0/params->w);
  params->pixel_dv = scl(vp_v, 1.0/params->h);
}



void render_to(unsigned char *pixels, RenderParams *params) {
#pragma acc update device(rp)
#pragma acc update device(n_objs, objs[:n_objs], nodes[:2*n_objs-1], images[:n_images])
  int rows_processed = 0;
#pragma acc kernels copy(pixels[:w*h*4]) copyin(rows_processed)

  // Main loop
#pragma acc loop independent
  for (int i = 0; i < params->h; i++) {

#pragma acc loop independent
    for (int j = 0; j < params->w; j++) {
      unsigned int X = i * params->w + j;
      v3 col = clampcol(compute_pixel_color(i, j, &X, params));
      int idx = 4 * (i*params->w + j);
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



void run(RenderParams *params) {
  assert(params->max_bounces <= MAX_MAX_BOUNCES);
  
  initialize_constants(params);
  if (n_objs > 0) build_lbvh();

  unsigned char *pixels = malloc(params->w * params->h * 4);
  render_to(pixels, params);
  stbi_write_png("output.png", params->w, params->h, 4, pixels, params->w * sizeof(unsigned int));
  free(pixels);
}



void cleanup() {
  // Free images
  for (int i = 0; i < n_images; i++) {
#if FOR_GPU == 0
    free(images[i].pixels);
#else
    cudaError_t err;
    if (err = cudaFree(images[i].pixels))
      printf("TEXTURE: failed to free image data -- %s\n", cudaGetErrorString(err));
#endif
  }
}

#endif