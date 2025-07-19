#include <stdlib.h>
#include "external/stb_image_write.h"
#include "render.c"
#include "scenes.c"

void setup_scene(RenderParams *params, RenderScene *scene, int scene_id) {
  // Default render parameters
  params->w = 1000;
  params->h = 700;
  params->max_bounces = 10;
  params->samples_per_pixel = 4;
  params->lookfrom = (v3){0,0,0};
  params->lookat = (v3){0,5,0};
  params->vfov = 45.0;
  params->defocus_angle = 0;

  scene->images = NULL;
  scene->prims = NULL;
  scene->nodes = NULL;
  if (scene_id == 0) setup_scene_debug(params, scene);
  else if (scene_id == 1) setup_scene_many_spheres(params, scene);
  else if (scene_id == 2) setup_scene_textures(params, scene);
  else if (scene_id == 3) setup_scene_quads(params, scene);
  else if (scene_id == 4) setup_scene_sdf(params, scene);
  else if (scene_id == 5) setup_scene_model(params, scene);
  else if (scene_id == 6) setup_scene_materials(params, scene);

  //// Derive remaining params ----------------------------------------------------

  params->focal_len = dist(params->lookat, params->lookfrom);
  params->defocus_radius = 2 * params->focal_len * tanf(params->defocus_angle/360*2*PI / 2);

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

void render_scene(RenderParams *params, RenderScene *scene) {
  assert(params->max_bounces <= MAX_MAX_BOUNCES);
  if (arrlen(scene->prims) > 0) build_lbvh(scene);
  unsigned char *pixels = (unsigned char*)malloc(params->w * params->h * 4);
  render_to_pixels(params, scene, pixels);
  printf("[rayt] Writing to output/out.png\n");
  stbi_write_png("output/out.png", params->w, params->h, 4, pixels, params->w * sizeof(unsigned int));
  free(pixels);
}

void cleanup_scene(RenderScene *scene) {
  if (scene->images) {
    for (int i = 0; i < arrlen(scene->images); i++)
      free(scene->images[i].pixels);
    free(scene->images);
  }
  // TODO: FIX THIS
  //if (scene->prims) free(scene->prims);
  if (scene->nodes) free(scene->nodes);
}

int main(int argc, char **argv) {
  RenderParams params;
  RenderScene scene;
  int scene_id;

  if (argc == 1) scene_id = 0;
  else if (argc > 1) scene_id = atoi(argv[1]);

  if (scene_id < 0 || scene_id > 6) scene_id = 0;

  printf("[rayt] Scene ID = %d ", scene_id);
  if (scene_id == 0) printf("(materials)\n");
  else if (scene_id == 1) printf("(many spheres)\n");
  else if (scene_id == 2) printf("(textures)\n");
  else if (scene_id == 3) printf("(quads)\n");
  else if (scene_id == 4) printf("(sdf)\n");
  else if (scene_id == 5) printf("(models)\n");
  else if (scene_id == 6) printf("(materials)\n");

  setup_scene(&params, &scene, scene_id);

  printf("[rayt] Render params: sky_color = (%.2f,%.2f,%.2f)\n", params.sky_color.x, params.sky_color.y, params.sky_color.z);
  printf("[rayt] Render params: size = %dx%d\n", params.h, params.w);
  printf("[rayt] Render params: max_bounces = %d\n", params.max_bounces);
  printf("[rayt] Render params: samples_per_pixel = %d\n", params.samples_per_pixel);
  printf("[rayt] Render params: lookfrom = (%.2f,%.2f,%.2f)\n", params.lookfrom.x, params.lookfrom.y, params.lookfrom.z);
  printf("[rayt] Render params: lookat = (%.2f,%.2f,%.2f)\n", params.lookat.x, params.lookat.y, params.lookat.z);
  printf("[rayt] Render params: vfov = %.2f degrees\n", params.vfov);
  printf("[rayt] Render params: defocus_angle = %.2f\n", params.defocus_angle);
  printf("[rayt] Render params: focal_len = %.2f\n", params.focal_len);
  printf("[rayt] Render params: defocus_radius = %.2f\n", params.defocus_radius);
  printf("[rayt] Render params: viewport_topleft = (%.2f,%.2f,%.2f)\n", params.viewport_topleft.x, params.viewport_topleft.y, params.viewport_topleft.z);
  printf("[rayt] Render params: pixel_du = (%.2f,%.2f,%.2f)\n", params.pixel_du.x, params.pixel_du.y, params.pixel_du.z);
  printf("[rayt] Render params: pixel_dv = (%.2f,%.2f,%.2f)\n", params.pixel_dv.x, params.pixel_dv.y, params.pixel_dv.z);
  printf("[rayt] Render params: cam_u = (%.2f,%.2f,%.2f)\n", params.cam_u.x, params.cam_u.y, params.cam_u.z);
  printf("[rayt] Render params: cam_v = (%.2f,%.2f,%.2f)\n", params.cam_v.x, params.cam_v.y, params.cam_v.z);

  render_scene(&params, &scene);
  cleanup_scene(&scene);
  return 0;
}
