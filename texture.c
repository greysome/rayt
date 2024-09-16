#ifndef _TEXTURE_C
#define _TEXTURE_C

#if FOR_GPU == 1
#include <cuda_runtime.h>
#endif

#include <stdlib.h>
#include "external/stb_image.h"
#include "vector.c"
#include "scene.c"


#define NO_IMAGE -1


Texture solid(v3 color) {
  Texture tex;
  tex.type = SOLID;
  tex.color = color;
  return tex;
}


Texture checker_abs(float size, v3 color_even, v3 color_odd) {
  Texture tex;
  tex.type = CHECKER_ABS;
  tex.size = size;
  tex.color_even = color_even;
  tex.color_odd = color_odd;
  return tex;
}


Texture checker_rel(float size, v3 color_even, v3 color_odd) {
  Texture tex;
  tex.type = CHECKER_REL;
  tex.size = size;
  tex.color_even = color_even;
  tex.color_odd = color_odd;
  return tex;
}






int load_image(RenderScene *scene, const char *img_path) {
  int w,h,n;
  unsigned char *pixels;
  pixels = stbi_load(img_path, &w, &h, &n, 3);
  if (pixels == NULL) {
    printf("TEXTURE: '%s' -- %s\n", img_path, stbi_failure_reason());
    return NO_IMAGE;
  }
  printf("TEXTURE: loaded %s (w=%d, h=%d, n=%d, p=%p)\n", img_path, w, h, n, pixels);
  
  Image img;
  img.w = w;
  img.h = h;
  img.pixels = pixels;
  arrpush(scene->images, img);
  return arrlen(scene->images) - 1;
}


Texture image_texture(int id) {
  Texture tex;
  tex.type = IMAGE;
  tex.image_id = id;
  return tex;
}






int clampi(int x, int low, int high) {
  if (x < low) return low;
  if (x > high) return high;
  return x;
}


v3 image_pixel_at(RenderScene *scene, Texture tex, float u, float v) {
  if (tex.type != IMAGE)
    return BLACK;
  Image img = scene->images[tex.image_id];
  int x = clampi(floorf(u * img.w), 0, img.w-1);
  int y = clampi(floorf(v * img.h), 0, img.h-1);

  int offset = 3*(y*img.w + x);
  float r = img.pixels[offset] / 255.0;
  float g = img.pixels[offset+1] / 255.0;
  float b = img.pixels[offset+2] / 255.0;
  return (v3){r,g,b};
}


static inline v3 get_texture_color(RenderScene *scene, Texture tex, float u, float v, v3 p) {
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
    return image_pixel_at(scene, tex, u, v);
  }
  return BLACK;
}

#endif