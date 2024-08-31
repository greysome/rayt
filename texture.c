#ifndef _TEXTURE_C
#define _TEXTURE_C

#if FOR_GPU == 1
#include <cuda_runtime.h>
#endif

#include <stdlib.h>
#include "external/stb_image.h"
#include "vector.c"

typedef enum {
  SOLID, CHECKER_ABS, CHECKER_REL, IMAGE
} TextureType;

typedef struct {
  int w, h;
  unsigned char *pixels;
} Image;

typedef struct {
  TextureType type;
  union {
    /* For solids */
    v3 color;

    /* For checker */
    struct {
      float size;
      v3 color_even;
      v3 color_odd;
    };

    /* For image */
    int image_id;
  };
} Texture;


#define MAX_IMAGES 1000
#define NO_IMAGE -1

int n_images = 0;
Image images[MAX_IMAGES];
#pragma acc declare create(images[:MAX_IMAGES])


Texture solid(v3 color) {
  return (Texture) {.type = SOLID, .color = color};
}

Texture checker_abs(float size, v3 color_even, v3 color_odd) {
  return (Texture) {.type = CHECKER_ABS, .size = size, .color_even = color_even, .color_odd = color_odd };
}

Texture checker_rel(float size, v3 color_even, v3 color_odd) {
  return (Texture) {.type = CHECKER_REL, .size = size, .color_even = color_even, .color_odd = color_odd };
}

int load_image(const char *img_path) {
  int w,h,n;
  unsigned char *pixels;
  pixels = stbi_load(img_path, &w, &h, &n, 3);
  if (pixels == NULL) {
    printf("TEXTURE: '%s' -- %s\n", img_path, stbi_failure_reason());
    return NO_IMAGE;
  }
  printf("TEXTURE: loaded %s (w=%d, h=%d, n=%d, p=%p)\n", img_path, w, h, n, pixels);
#if FOR_GPU == 0
  images[n_images++] = (Image) {.w = w, .h = h, .pixels = pixels};
  return n_images-1;
#else
  // Transfer image texture data to the GPU
  unsigned char *pixels_device;
  cudaError_t err;
  if (err = cudaMalloc((void **) &pixels_device, 3*w*h))
    printf("TEXTURE: failed to allocate memory on device for image data -- %s\n", cudaGetErrorString(err));
  if (err = cudaMemcpy(pixels_device, pixels, 3*w*h, cudaMemcpyHostToDevice))
    printf("TEXTURE: failed to transfer image data from host to device -- %s\n", cudaGetErrorString(err));
  free(pixels);
  images[n_images++] = (Image) {.w = w, .h = h, .pixels = pixels_device};
  return n_images-1;
#endif
}

Texture image(int id) {
  return (Texture) {.type = IMAGE, .image_id = id };
}

int clampi(int x, int low, int high) {
  if (x < low) return low;
  if (x > high) return high;
  return x;
}

v3 image_pixel_at(Texture tex, float u, float v) {
  if (tex.type != IMAGE)
    return BLACK;
  Image img = images[tex.image_id];
  int x = clampi(floorf(u * img.w), 0, img.w-1);
  int y = clampi(floorf(v * img.h), 0, img.h-1);

  int offset = 3*(y*img.w + x);
  float r = img.pixels[offset] / 255.0;
  float g = img.pixels[offset+1] / 255.0;
  float b = img.pixels[offset+2] / 255.0;
  return (v3){r,g,b};
}

static inline v3 get_texture_color(Texture tex, float u, float v, v3 p) {
  if (tex.type == SOLID)
    return tex.color;
  else if (tex.type == CHECKER_ABS) {
    int xx = floorf(p.x / tex.size);
    int yy = floorf(p.y / tex.size);
    int zz = floorf(p.z / tex.size);
    return (xx+yy+zz) % 2 == 0 ? tex.color_even : tex.color_odd;
  }
  else if (tex.type == CHECKER_REL) {
    int xx = floorf(u / tex.size);
    int yy = floorf(v / tex.size);
    return (xx+yy) % 2 == 0 ? tex.color_even : tex.color_odd;
  }
  else if (tex.type == IMAGE) {
    return image_pixel_at(tex, u, v);
  }
  return BLACK;
}

#endif