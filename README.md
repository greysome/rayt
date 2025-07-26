# rayt

(Final scene from *Ray Tracing in a Weekend*, rendered in 0.1s on an NVIDIA TITAN V)
![spheres](/readme_assets/spheres.jpg)

A raytracer written in C and CUDA. My eventual goal is to efficiently
render a set of [benchmark
scenes](https://benedikt-bitterli.me/resources/), though it might take
a lot of work. I write from scratch as much as possible, including the
parsers for asset files.

Some features:
- GPU-specific optimisations like LBVH
- Hand-written .obj and .ply parser

Features I want to include:
- .pbrt/.pfm/.tga parsers
- Incorporate more sophisticated rendering techniques (i.e. read
  through pbrt and research papers)