#include "render.c"

void demo_materials() {
  sky_color = (v3){0.7,0.9,1};
  //add_sphere((v3){0,-5000,0}, 5000, dielectric((v3){1,0.6,0.6}, 1.3));
  add_sphere((v3){0,-5000,0}, 5000, matte(GREEN, 0.2));
  add_sphere((v3){-2,1,3}, 1, dielectric(WHITE, 1.3));
  add_sphere((v3){0,1,3}, 1, matte(RED, 0.2));
  add_sphere((v3){2,1,3}, 1, metal((v3){0.6,0.6,1}, 1, 0.2));
  add_sphere((v3){0,3,3}, 0.5, light((v3){5,5,5}));
  lookfrom = (v3){0,1.5,-3};
  lookat = (v3){0,1,3};
  num_samples = 1000;
}

void demo_many_spheres() {
  sky_color = (v3){0.7,0.9,1};
  add_sphere((v3){0,-1000,0}, 1000, matte(scl(WHITE,0.5), 0.5));

  unsigned int X = 3;

  for (int i = -11; i < 11; i++) {
    for (int j = -11; j < 11; j++) {
      v3 center = {i+0.8*randunif(&X), 0.2, j+0.8*randunif(&X)};
      if (dist(center, (v3){4,0.2,0}) <= 0.9)
      	continue;

      float choose_mat = randunif(&X);
      float r = randunif(&X);
      float g = randunif(&X);
      float b = randunif(&X);
      v3 col = mul((v3){r,g,b},(v3){r,g,b});
      if (choose_mat < 0.8) {
	add_sphere(center, 0.2, matte(col, 0.5));
      }
      else if (choose_mat < 0.95) {
	r = r/2.0 + 0.5;
	g = g/2.0 + 0.5;
	b = b/2.0 + 0.5;
	add_sphere(center, 0.2, metal((v3){r,g,b}, 0.7, 0.2));
      }
      else {
	add_sphere(center, 0.2, dielectric(WHITE, 1.5));
      }
    }
  }

  add_sphere((v3){0,1,0}, 1, dielectric(WHITE,1.5));
  add_sphere((v3){4,1,0}, 1, metal((v3){0.8,0.6,0.6}, 1, 0));
  add_sphere((v3){-4,1,0}, 1, matte((v3){0.4,0.2,0.1}, 0.5));

  max_recurse = 10;
  num_samples = 500;
  width = 1500;
  height = 1000;
  vfov = 30;
  lookfrom = (v3){13,2,-3};
  lookat = (v3){0,0,0};
  defocus_angle = 0;
}

void setup_scene() {
  int i = 0;
  switch (i) {
  case 0: demo_materials(); break;
  case 1: demo_many_spheres(); break;
  default: demo_materials(); break;
  }
  stack_size = 1024;
}

int main() {
  setup_scene();
  run();
  return 0;
}