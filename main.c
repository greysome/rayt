#include "render.c"

void demo_materials() {
  sky_color = (v3){.7,.9,1};
  //add_sphere((v3){0,-5000,0}, 5000, dielectric((v3){1,0.6,0.6}, 1.3));
  add_sphere((v3){0,-5000,0}, 5000, matte(0.2), solid(GREEN));
  add_sphere((v3){-2,1,3}, 1, dielectric(1.3), solid(WHITE));
  add_sphere((v3){0,1,3}, 1, matte(0.2), solid(RED));
  add_sphere((v3){2,1,3}, 1, metal(1, 0.2), solid((v3){.6,.6,1}));
  add_sphere((v3){0,3,3}, 0.5, light(), solid(scl(WHITE,3)));
  lookfrom = (v3){0,1.5,-3};
  lookat = (v3){0,1,3};
  num_samples = 4;
}

void demo_many_spheres() {
  sky_color = (v3){.7,.9,1};
  add_sphere((v3){0,-1000,0}, 1000, matte(0.5), solid(scl(WHITE,0.5)));

  unsigned int X = 3;

  for (int i = -11; i < 11; i++) {
    for (int j = -11; j < 11; j++) {
      v3 center = {i+0.8*randunif(&X), 0.2, j+0.8*randunif(&X)};
      if (dist(center, (v3){4,.2,0}) <= 0.9)
      	continue;

      float choose_mat = randunif(&X);
      float r = randunif(&X);
      float g = randunif(&X);
      float b = randunif(&X);
      v3 col = mul((v3){r,g,b},(v3){r,g,b});
      if (choose_mat < 0.8) {
	add_sphere(center, 0.2, matte(0.5), solid(col));
      }
      else if (choose_mat < 0.95) {
	r = r/2.0 + 0.5;
	g = g/2.0 + 0.5;
	b = b/2.0 + 0.5;
	add_sphere(center, 0.2, metal(.7,.2), solid((v3){r,g,b}));
      }
      else {
	add_sphere(center, 0.2, dielectric(1.5), solid(WHITE));
      }
    }
  }

  add_sphere((v3){0,1,0}, 1, dielectric(1.5), solid(WHITE));
  add_sphere((v3){4,1,0}, 1, metal(1, 0), solid((v3){.8,.6,.6}));
  add_sphere((v3){-4,1,0}, 1, matte(0.5), solid((v3){.4,.2,.1}));

  max_recurse = 10;
  num_samples = 500;
  width = 1000;
  height = 700;
  vfov = 20;
  lookfrom = (v3){13,2,-3};
  lookat = (v3){0,0,0};
  defocus_angle = 0;
}

void demo_textures() {
  width = 700;
  height = 500;
  sky_color = (v3){.6,.8,1};
  add_sphere((v3){0,-1000,0}, 1000, matte(0), checker_abs(1, WHITE, GRAY));
  add_sphere((v3){-1,1,1}, 1, matte(0), image("assets/earthmap.jpg"));
  //add_sphere((v3){-1,1,1}, 1, matte(0.5), checker_rel(0.25, BLUE, RED));
  add_sphere((v3){1,1,1}, 1, dielectric(1.5), solid(WHITE));
  lookfrom = (v3){0,2,-7};
  lookat = (v3){0,0,5};
  vfov = 30;
  num_samples = 10;
}

void setup_scene() {
  int i = 2;
  switch (i) {
  case 0: demo_materials(); break;
  case 1: demo_many_spheres(); break;
  case 2: demo_textures(); break;
  default: demo_materials(); break;
  }
  stack_size = 1024;
}

int main() {
  setup_scene();
  run();
  return 0;
}