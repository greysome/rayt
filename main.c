#include "render.c"

#ifndef SCENEIDX
#define SCENEIDX 0
#endif
RenderParams params;


void demo_materials() {
  params.w = 300;
  params.h = 300;
  params.sky_color = (v3){.7,.9,1};
  params.lookfrom = (v3){0,1.5,-3};
  params.lookat = (v3){0,1,3};
  params.samples_per_pixel = 4;

  //add_sphere((v3){0,-5000,0}, 5000, dielectric((v3){1,0.6,0.6}, 1.3));
  add_sphere((v3){0,-5000,0}, 5000, matte(), solid(GREEN));
  add_sphere((v3){-2,1,3}, 1, dielectric(1.3), solid(WHITE));
  add_sphere((v3){0,1,3}, 1, matte(), solid(RED));
  add_sphere((v3){2,1,3}, 1, metal(1, 0.2), solid((v3){.6,.6,1}));
  add_sphere((v3){0,3,3}, 0.5, light(), solid(scl(WHITE,3)));
}



void demo_many_spheres() {
  params.w = 1000;
  params.h = 700;
  params.sky_color = (v3){.7,.9,1};
  params.max_bounces = 10;
  params.samples_per_pixel = 500;
  params.vfov = 20;
  params.lookfrom = (v3){13,2,-3};
  params.lookat = (v3){0,0,0};
  params.defocus_angle = 0;

  add_sphere((v3){0,-1000,0}, 1000, matte(), solid(scl(WHITE,0.5)));

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
	add_sphere(center, 0.2, matte(), solid(col));
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
  add_sphere((v3){-4,1,0}, 1, matte(), solid((v3){.4,.2,.1}));
}



void demo_textures() {
  params.w = 700;
  params.h = 500;
  params.sky_color = (v3){.6,.8,1};
  params.lookfrom = (v3){0,2,-7};
  params.lookat = (v3){0,0,5};
  params.vfov = 30;
  params.samples_per_pixel = 4;

  add_sphere((v3){0,-1000,0}, 1000, matte(), checker_abs(1, WHITE, GRAY));
  add_sphere((v3){-1,1,1}, 1, matte(), image(load_image("assets/earthmap.jpg")));
  //add_sphere((v3){-1,1,1}, 1, matte(0.5), checker_rel(0.25, BLUE, RED));
  add_sphere((v3){1,1,1}, 1, dielectric(1.5), solid(WHITE));
}



void demo_quads() {
  params.w = 300;
  params.h = 300;
  params.sky_color = (v3){.6,.8,1};

  params.samples_per_pixel = 20;
  params.max_bounces = 10;
  params.vfov = 80;
  params.lookfrom = (v3){0,0,9};
  params.lookat = (v3){0,0,0};
  params.defocus_angle = 0;

  add_quad((v3){-3,-2,5}, (v3){0,0,-4}, (v3){0,4,0}, matte(), solid((v3){1,.2,.2}));
  add_quad((v3){-2,-2,0}, (v3){4,0,0}, (v3){0,4,0}, matte(), solid((v3){.2,1,.2}));
  add_quad((v3){3,-2,1}, (v3){0,0,4}, (v3){0,4,0}, matte(), solid((v3){.2,.2,1}));
  add_quad((v3){-2,3,1}, (v3){4,0,0}, (v3){0,0,4}, matte(), solid((v3){1,.5,0}));
  add_quad((v3){-2,-3,5}, (v3){4,0,0}, (v3){0,0,-4}, matte(), solid((v3){.2,.8,.8}));
}



float sdf1(v3 p) {
  return sqrtf(0.3*p.x*p.x + p.y*p.y + p.z*p.z) - 1;
}
aabb sdf1_aabb = {(v3){-2,-1,-1}, (v3){2,1,1}};

void demo_sdf() {
  params.w = 500;
  params.h = 300;
  params.sky_color = (v3){.6,.8,1};

  params.samples_per_pixel = 100;
  params.max_bounces = 10;
  params.vfov = 60;
  params.lookfrom = (v3){0,2,-6};
  params.lookat = (v3){0,1,0};
  params.defocus_angle = 0;

  // Ground
  add_quad((v3){-1000,0,-1000}, (v3){2000,0,0}, (v3){0,0,2000}, matte(), solid(GREEN));
  add_sdf((v3){0,1,0}, 1, matte(), solid(RED));
}



void demo_model() {
  params.sky_color = (v3){0,0,0};
  params.w = 1000;
  params.h = 700;
  params.vfov = 60;
  params.lookfrom = (v3){-10,40,0};
  params.lookat = (v3){0,0,0};
  params.samples_per_pixel = 200;
  
  add_sphere((v3){8,20,-5}, 10, light(), solid((v3){2,0.8,0.8}));
  add_sphere((v3){-10,10,15}, 5, light(), solid((v3){0.8,2,0.8}));
  add_sphere((v3){-12,14,-12}, 7, light(), solid((v3){0.8,0.8,2}));
  add_quad((v3){-1000,0,-1000}, (v3){2000,0,0}, (v3){0,0,2000}, matte(), solid((v3){.5,.5,.5}));

  Face *faces = load_obj("box.obj");
  arrfree(faces);
}




RenderParams setup_scene() {
  // Set some default render parameters
  params.w = 1000;
  params.h = 700;
  params.max_bounces = 10;
  params.samples_per_pixel = 4;
  params.lookfrom = (v3){0,0,0};
  params.lookat = (v3){0,5,0};
  params.vfov = 45.0;
  params.defocus_angle = 0;

  switch (SCENEIDX) {
  case 0: demo_materials(); break;
  case 1: demo_many_spheres(); break;
  case 2: demo_textures(); break;
  case 3: demo_quads(); break;
  case 4: demo_sdf(); break;
  case 5: demo_model(); break;
  default: demo_materials(); break;
  }
  return params;
}



int main() {
  setup_scene();
  run(&params);
  cleanup();
  return 0;
}