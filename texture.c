#ifndef _TEXTURE_C
#define _TEXTURE_C

#if FOR_GPU == 1
#include <cuda_runtime.h>
#endif

#include "stb_image.h"
#include <stdlib.h>
#include "vector.c"

typedef enum {
  SOLID, CHECKER_ABS, CHECKER_REL, IMAGE
} TextureType;

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
    struct {
      int w, h;
      unsigned char *pixels;
    };
  };
} Texture;

Texture solid(v3 color) {
  return (Texture) {.type = SOLID, .color = color};
}

Texture checker_abs(float size, v3 color_even, v3 color_odd) {
  return (Texture) {.type = CHECKER_ABS, .size = size, .color_even = color_even, .color_odd = color_odd };
}

Texture checker_rel(float size, v3 color_even, v3 color_odd) {
  return (Texture) {.type = CHECKER_REL, .size = size, .color_even = color_even, .color_odd = color_odd };
}

Texture image(const char *img_path) {
  int w,h,n;
  unsigned char *pixels;
  pixels = stbi_load(img_path, &w, &h, &n, 3);
  if (pixels == NULL) {
    printf("TEXTURE: '%s' -- %s\n", img_path, stbi_failure_reason());
    return solid((v3){0,1,1});
  }
  printf("TEXTURE: loaded %s, (w=%d,h=%d,n=%d,p=%p)\n", img_path, w, h, n, pixels);
  // Transfer image texture data to the GPU
#if FOR_GPU == 0
  return (Texture) {.type = IMAGE, .w = w, .h = h, .pixels = pixels};
#else
  unsigned char *pixels_device;
  cudaError_t err;
  if (err = cudaMalloc((void **) &pixels_device, 3*w*h))
    printf("TEXTURE: failed to allocate memory on device for image data -- %s\n", cudaGetErrorString(err));
  if (err = cudaMemcpy(pixels_device, pixels, 3*w*h, cudaMemcpyHostToDevice))
    printf("TEXTURE: failed to transfer image data from host to device -- %s\n", cudaGetErrorString(err));
  free(pixels);
  return (Texture) {.type = IMAGE, .w = w, .h = h, .pixels = pixels_device};
#endif
}

int clampi(int x, int low, int high) {
  if (x < low) return low;
  if (x > high) return high;
  return x;
}

v3 image_pixel_at(Texture tex, float u, float v) {
  if (tex.type != IMAGE)
    return BLACK;
  int x = clampi(floorf(u * tex.w), 0, tex.w-1);
  int y = clampi(floorf(v * tex.h), 0, tex.h-1);

  int offset = 3*(y*tex.w + x);
  float r = tex.pixels[offset] / 255.0;
  float g = tex.pixels[offset+1] / 255.0;
  float b = tex.pixels[offset+2] / 255.0;
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