#ifndef _RENDER_C
#define _RENDER_C

#include <math.h>
#include <sys/time.h>
#include <stdio.h>
#include "common.h"
#include "texture.c"
#include "material.c"
#include "primitive.c"
#include "lbvh.c"

// The maximum that RenderParams.max_bounces can be, lol
// The reason is because I originally wanted a VLA with size
// RenderParams.max_bounces in get_color(), but this is not suppoted
// by pgcc...
#define MAX_MAX_BOUNCES 64
// Blocking factor for render_kernel()
#define NB 16

__device__ v3 gamma_correct(v3 col) {
  return (v3) {sqrtf(col.x), sqrtf(col.y), sqrtf(col.z)};
}

__device__ v3 shoot_ray(RenderParams *params, RenderScene *scene, v3 origin, v3 dir, unsigned int *X) {
  v3 cur_origin = origin;
  v3 cur_dir = dir;
  v3 cur_color = WHITE;

  // Imitating a stack. I don't use recursion because of the GPU's small stack.
  int depth = 0;
  Material materials[MAX_MAX_BOUNCES];
  v3 tex_colors[MAX_MAX_BOUNCES];

  for (; depth < params->max_bounces; depth++) {
    // Obtain closest intersection
    HitRecord hr;
    if (scene->prims != NULL)
      get_closest_intersection(scene, cur_origin, cur_dir, &hr);
    else
      hr.t = INFINITY;

    // If it is the sky, the rays stops bouncing and we proceed to mix together
    // all the color contributions
    if (hr.t == INFINITY) {
      cur_color = params->sky_color;
      depth--;
      goto fold_stack;
    }

    // Add material and texture color to the stack
    materials[depth] = hr.prim.mat;
    tex_colors[depth] = get_texture_color(scene, hr.prim.tex, hr.u, hr.v, hr.p);

    v3 normal = get_normal(hr.prim, hr.p);
    v3 out_dir;
    interact_with_material(hr.prim.mat, normal, cur_dir, &out_dir, X);

    // Offset intersection point slightly to prevent shadow acne
    hr.p = ray_at(hr.p, out_dir, 0.01);
    cur_origin = hr.p;
    cur_dir = out_dir;

    // If it is a light source, rays also stops bouncing
    // The light's material is already added onto the material stack, so we
    // don't have to set cur_color
    if (hr.prim.mat.type == LIGHT)
      goto fold_stack;
  }
  goto fold_stack;

fold_stack:
  for (; depth >= 0; depth--)
    cur_color = get_reflected_color(materials[depth], tex_colors[depth], cur_color);
  return cur_color;
}

__device__ v3 compute_pixel_color(RenderParams *params, RenderScene *scene, int i, int j, unsigned int *X) {
  v3 avg_col = {0,0,0};
  for (int k = 0; k < params->samples_per_pixel; k++) {
    // Jitter the pixel a little for randomness
    v2 pix_offset = randsquare(X);
    // Convert 2D pixel coords to 3D scene coords
    v3 pix_pos = add(params->viewport_topleft,
                     add(scl(params->pixel_du, (j + 0.5 + pix_offset.x)),
                         scl(params->pixel_dv, (i + 0.5 + pix_offset.y))));

    v2 defocus_offset = scl2(randdisk(X), params->defocus_radius);
    v3 defocused_lookfrom = add(params->lookfrom,
				                        add(scl(params->cam_u, defocus_offset.x),
				                            scl(params->cam_v, defocus_offset.y)));
    v3 dir = sub(pix_pos, defocused_lookfrom);

    v3 ray_col = shoot_ray(params, scene, defocused_lookfrom, dir, X);
    avg_col = add(avg_col, ray_col);
  }
  avg_col = gamma_correct(scl(avg_col, 1.0/params->samples_per_pixel));
  return avg_col;
}

__global__ void render_kernel(RenderParams *params, RenderScene *scene, unsigned char *pixels) {
  int i = blockIdx.x * NB + threadIdx.x;
  int j = blockIdx.y * NB + threadIdx.y;
  //if (i > 0 || j > 0) return;
  if (i >= params->h || j >= params->w) return;

  // Initial seed for RNG
  unsigned int X = i * params->w + j;
  v3 col = clampcol(compute_pixel_color(params, scene, i, j, &X));
  int idx = 4 * (i * params->w + j);
  pixels[idx]   = (unsigned char)(col.x * 255);
  pixels[idx+1] = (unsigned char)(col.y * 255);
  pixels[idx+2] = (unsigned char)(col.z * 255);
  pixels[idx+3] = 255;
}

void copy_scene_to_device(RenderScene *d_scene, RenderScene *scene) {
  // host_d_* means a host-side copy of a structure that contains
  // device pointers, to be cudaMemcpy-ed to the device proper.

  int images_size = arrlen(scene->images) * sizeof(Image);
  // Host-side array of device pointers
  Image *host_d_images = (Image*) malloc(sizeof(images_size));
  // Device-side array of device pointers
  Image *d_images;
  cudaMalloc((void**) &d_images, images_size);
  CUDA_CATCH();

  for (int i = 0; i < arrlen(scene->images); i++) {
    // Copy pixel data to device
    Image image = scene->images[i];
    unsigned char *d_pixels;
    cudaMalloc((void**) &d_pixels, image.w * image.h * 4);
    CUDA_CATCH();
    cudaMemcpy(d_pixels, image.pixels, image.w * image.h * 4, cudaMemcpyHostToDevice);
    CUDA_CATCH();

    // Append image to host-side array
    Image host_d_image;
    host_d_image.w = image.w;
    host_d_image.h = image.h;
    host_d_image.pixels = d_pixels;
    host_d_images[i] = host_d_image;
  }
  // Populate device pointers in device-side array
  cudaMemcpy(d_images, host_d_images, images_size, cudaMemcpyHostToDevice);
  CUDA_CATCH();

  int prims_size = arrlen(scene->prims) * sizeof(Primitive);
  Primitive *d_prims;
  cudaMalloc((void**) &d_prims, prims_size);
  CUDA_CATCH();
  cudaMemcpy(d_prims, scene->prims, prims_size, cudaMemcpyHostToDevice);
  CUDA_CATCH();

  int nodes_size = (2 * arrlen(scene->prims) - 1) * sizeof(lbvh_node);
  lbvh_node *d_nodes;
  cudaMalloc((void**) &d_nodes, nodes_size);
  CUDA_CATCH();
  cudaMemcpy(d_nodes, scene->nodes, nodes_size, cudaMemcpyHostToDevice);
  CUDA_CATCH();

  RenderScene host_d_scene;
  host_d_scene.images = d_images;
  host_d_scene.prims = d_prims;
  host_d_scene.nodes = d_nodes;
  cudaMemcpy(d_scene, &host_d_scene, sizeof(RenderScene), cudaMemcpyHostToDevice);
  CUDA_CATCH();
}

void render_to_pixels(RenderParams *params, RenderScene *scene, unsigned char *pixels) {
  RenderParams *d_params;
  RenderScene *d_scene;
  unsigned char *d_pixels;

  cudaMalloc((void**) &d_params, sizeof(RenderParams));
  CUDA_CATCH();
  cudaMalloc((void**) &d_scene, sizeof(RenderScene));
  CUDA_CATCH();
  cudaMalloc((void**) &d_pixels, params->h * params->w * 4);
  CUDA_CATCH();

  cudaMemcpy(d_params, params, sizeof(RenderParams), cudaMemcpyHostToDevice);
  CUDA_CATCH();
  copy_scene_to_device(d_scene, scene);

#define CEILDIV(x,y) (((x)+(y)-1)/(y))
  dim3 grid_dim(CEILDIV(params->h, NB), CEILDIV(params->w, NB));
  dim3 block_dim(NB, NB);
  printf("[rayt] Lauching render kernel: grid=%dx%d block=%dx%d\n", params->h/NB, params->w/NB, NB, NB);

  struct timeval tv_start, tv_end;
  gettimeofday(&tv_start, NULL);
  render_kernel<<<grid_dim, block_dim>>>(d_params, d_scene, d_pixels);
  cudaDeviceSynchronize();
  gettimeofday(&tv_end, NULL);
  CUDA_CATCH();

  int seconds = (tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec);
  printf("[rayt] Render kernel took %.2f seconds\n", (float)seconds/1000000);

  cudaMemcpy(pixels, d_pixels, params->h * params->w * 4, cudaMemcpyDeviceToHost);
  CUDA_CATCH();
  cudaFree(d_params);
  CUDA_CATCH();
  cudaFree(d_scene);
  CUDA_CATCH();
  cudaFree(d_pixels);
  CUDA_CATCH();
}

#endif
